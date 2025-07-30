#![deny(unused_imports)]

use anyhow::Result;
use brainwhat::{instruction::Instruction, optimizer, parser};
use crossbeam_channel::unbounded;
use dashmap::DashMap;
use indicatif::{ProgressBar, ProgressStyle};
use once_cell::sync::Lazy;
use rand::{
    distributions::{Distribution, WeightedIndex, Standard},
    prelude::*,
    thread_rng,
};
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use std::{
    collections::VecDeque,
    fs::OpenOptions,
    io::{BufWriter, Write},
    thread,
};

use clap::Parser;

///  GA search for novel Brain‑what programs
#[derive(Parser)]
struct Args {
    /// Path of the archive file to append to *and* resume from
    #[arg(long, default_value = "programs.txt")]
    programs: String,
}



/* ---------- global knobs ------------------------ */

const POP_SIZE: usize      = 4*4_096;
const GENERATIONS: usize   = 8*200_000;    // evaluation budget = POP_SIZE * GENERATIONS
const ELITES: usize        = 128;         // kept unchanged each generation
const MAX_CODE_LEN: usize  = 16;
const STEP_CAP: usize      = 5_000;

const WEIGHT_N: f32 = 0.7;
const WEIGHT_L: f32 = 0.3;

/* ----------- random program generator ----------- */

const MAX_DEPTH: usize = 8;
const WEIGHTED_OPS: &[(char, u8)] = &[('+', 6), ('-', 6), ('>', 3), ('<', 3), ('.', 4)];
static OP_DIST: Lazy<WeightedIndex<u8>> =
    Lazy::new(|| WeightedIndex::new(WEIGHTED_OPS.iter().map(|&(_, w)| w)).unwrap());

const LEN_DIST: &[(usize, u8)] = &[(4, 30), (5, 25), (6, 20), (7, 15), (8, 10)];
static LEN_W: Lazy<WeightedIndex<u8>> =
    Lazy::new(|| WeightedIndex::new(LEN_DIST.iter().map(|&(_, w)| w)).unwrap());

fn sample_op<R: Rng + ?Sized>(rng: &mut R) -> char {
    WEIGHTED_OPS[OP_DIST.sample(rng)].0
}

fn random_program<R: Rng>(rng: &mut R) -> String {
    let len = LEN_DIST[LEN_W.sample(rng)].0;
    let mut prog = Vec::with_capacity(len);
    let mut depth = 0usize;
    let mut printed = false;
    let mut i = 0usize;

    while i < len {
        let remaining = len - i;

        if depth > 0 && remaining == depth {
            prog.push(']');
            depth -= 1;
            i += 1;
            continue;
        }
        if !printed && remaining == depth + 1 {
            prog.push('.');
            printed = true;
            i += 1;
            continue;
        }

        let target_depth = rng.gen_range(0..=MAX_DEPTH);
        let ch = if depth > target_depth && depth > 0 && rng.gen_bool(0.6) {
            depth -= 1;
            ']'
        } else if depth < target_depth && remaining > depth + 1 && rng.gen_bool(0.6) {
            depth += 1;
            '['
        } else {
            sample_op(rng)
        };

        if ch == '.' {
            printed = true;
        }
        prog.push(ch);
        i += 1;
    }
    prog.into_iter().collect()
}

/* ------------- evaluation helpers -------------- */

fn run_ok(code: &[Instruction]) -> Option<String> {
    let mut tape: VecDeque<u8> = VecDeque::from([0]);
    let mut ptr: isize = 0;
    let (mut pc, mut steps) = (0usize, 0usize);
    let mut out = Vec::<u8>::new();

    while pc < code.len() && steps < STEP_CAP {
        use Instruction::*;
        let mut advance = true;
        match code[pc] {
            Move(d) => {
                ptr += d as isize;
                while ptr < 0 {
                    tape.push_front(0);
                    ptr += 1;
                }
                while ptr as usize >= tape.len() {
                    tape.push_back(0);
                }
            }
            Add(v) => {
                let cell = &mut tape[ptr as usize];
                if v >= 0 {
                    *cell = cell.wrapping_add(v as u8);
                } else {
                    *cell = cell.wrapping_sub((-v) as u8);
                }
            }
            Clear => tape[ptr as usize] = 0,
            Print => {
                let v = tape[ptr as usize];
                if v != 0 {
                    out.push(v);
                }
            }
            Read => unreachable!(),
            JumpIfZero(t) => {
                if tape[ptr as usize] == 0 {
                    pc = t;
                }
            }
            JumpIfNonZero(t) => {
                if tape[ptr as usize] != 0 {
                    pc = t;
                    advance = false;
                }
            }
        }
        if advance {
            pc += 1;
        }
        steps += 1;
    }
    (pc == code.len()).then(|| {
        out.iter()
            .map(u8::to_string)
            .collect::<Vec<_>>()
            .join(" ")
    })
}

fn eval(src: &str) -> Option<String> {
    parser::parse(&src.chars().collect::<Vec<_>>())
        .ok()
        .and_then(|p| optimizer::optimize(&p).ok())
        .and_then(|ir| run_ok(&ir))
}

/* ------------- mutation & crossover ------------ */

const MUTATE_CHARS: &[u8] = b"><+-.";
#[derive(Copy, Clone)]
enum Mutation {
    Insert,
    Replace,
    Wrap,
    Delete,
}

impl Distribution<Mutation> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Mutation {
        match rng.gen_range(0..100) {
            0..=49 => Mutation::Insert,
            50..=79 => Mutation::Replace,
            80..=89 => Mutation::Wrap,
            _ => Mutation::Delete,
        }
    }
}

fn mutate_once<R: Rng>(rng: &mut R, src: &str) -> String {
    let mut v: Vec<char> = src.chars().collect();
    match rng.gen::<Mutation>() {
        Mutation::Insert => {
            let pos = rng.gen_range(0..=v.len());
            let ch = MUTATE_CHARS[rng.gen_range(0..MUTATE_CHARS.len())] as char;
            v.insert(pos, ch);
        }
        Mutation::Replace => {
            if !v.is_empty() {
                let pos = rng.gen_range(0..v.len());
                v[pos] = MUTATE_CHARS[rng.gen_range(0..MUTATE_CHARS.len())] as char;
            }
        }
        Mutation::Wrap => {
            let start = rng.gen_range(0..=v.len());
            v.insert(start, '[');
            let end = rng.gen_range(start + 1..=v.len());
            v.insert(end, ']');
        }
        Mutation::Delete => {
            if !v.is_empty() {
                let pos = rng.gen_range(0..v.len());
                v.remove(pos);
            }
        }
    }
    v.into_iter().collect()
}

fn crossover<R: Rng>(rng: &mut R, a: &str, b: &str) -> String {
    if a.is_empty() || b.is_empty() {
        return a.to_owned();
    }
    let (ac, bc): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let cut_a = rng.gen_range(0..=ac.len());
    let cut_b = rng.gen_range(0..=bc.len());
    ac[..cut_a]
        .iter()
        .chain(bc[cut_b..].iter())
        .copied()
        .collect()
}

/* --------------- GA data types ----------------- */

#[derive(Clone)]
struct Individual {
    code: String,
    output: String,
    fitness: f32,
}

impl Individual {
    fn dummy() -> Self {
        Self {
            code: String::new(),
            output: String::new(),
            fitness: f32::NEG_INFINITY,
        }
    }
}

fn fitness(output_seen: bool, code_len: usize) -> f32 {
    let novelty = if output_seen { 0.0 } else { 1.0 };
    let length_score = 1.0 / code_len as f32;
    WEIGHT_N * novelty + WEIGHT_L * length_score
}

/* -------------- tournament select ------------- */

fn tournament_select<'a, R: Rng>(rng: &mut R, pool: &'a [Individual]) -> &'a Individual {
    let k = 4;
    (0..k)
        .map(|_| &pool[rng.gen_range(0..pool.len())])
        .max_by(|x, y| x.fitness.partial_cmp(&y.fitness).unwrap())
        .unwrap()
}

/* -------------------- main -------------------- */

fn main() -> Result<()> {

    /* ---------- CLI ---------- */
    let args = Args::parse();
    let file_path = args.programs;          // single source of truth

    /* ---------- novelty archive ---------- */
    let archive: DashMap<String, ()> = DashMap::new();
    if let Ok(txt) = std::fs::read_to_string(&file_path) {
        for rec in txt.split("###\n").filter(|s| !s.trim().is_empty()) {
            let mut lines = rec.lines();
            let _code = lines.next();                           // line 1 – program string
            if let Some(out_line) = lines.next() {              // line 2 – "-> xx xx"
                if let Some(out) = out_line.strip_prefix("->") {
                    archive.insert(out.trim().to_owned(), ());
                }
            }
        }
    }

    /* ---------- writer thread ---------- */
    let (tx, rx) = unbounded::<String>();
    let writer_path = file_path.clone();
    let writer = thread::spawn(move || -> Result<()> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&writer_path)?;
        let mut w = BufWriter::new(file);
        for line in rx {
            w.write_all(line.as_bytes())?;
        }
        w.flush()?;
        Ok(())
    });
    /* shared novelty archive */
    let archive: DashMap<String, ()> = DashMap::new();

    /* progress bar */
    let pb = ProgressBar::new((POP_SIZE * GENERATIONS) as u64);
    pb.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len}{percent:>3}%")
            .unwrap(),
    );

    /* initialise population */
    let mut population: Vec<Individual> = (0..POP_SIZE)
        .into_par_iter()
        .map_init(
            || Pcg64Mcg::from_rng(thread_rng()).unwrap(),
            |rng, _| {
                let code = random_program(rng);
                let output = eval(&code).unwrap_or_default();
                let fit = fitness(false, code.len()); // archive empty at start
                Individual { code, output, fitness: fit }
            },
        )
        .collect();

    /* GA loop */
    for _gen in 0..GENERATIONS {
        /* evaluate & update fitness in parallel */
        population.par_iter_mut().for_each(|ind| {
            if ind.code.is_empty() {
                ind.fitness = f32::NEG_INFINITY;
                return;
            }
            let seen = archive.contains_key(&ind.output);
            ind.fitness = fitness(seen, ind.code.len());
        });

        /* sort descending by fitness */
        population.par_sort_unstable_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        /* record genuinely new outputs */
        for ind in &population {
            if archive
                .insert(ind.output.clone(), ())
                .is_none()
            {
                let _ = tx.send(format!("###\n{}\n-> {}\n\n", ind.code, ind.output));
            }
        }

        /* elitism */
        let elites: Vec<Individual> = population[..ELITES].to_vec();

        /* offspring */
        let offspring: Vec<Individual> = (ELITES..POP_SIZE)
            .into_par_iter()
            .map_init(
                || Pcg64Mcg::from_rng(thread_rng()).unwrap(),
                |rng, _| {
                    let p1 = tournament_select(rng, &population);
                    let p2 = tournament_select(rng, &population);
                    let mut child_code = crossover(rng, &p1.code, &p2.code);
                    if rng.gen_bool(0.8) {
                        child_code = mutate_once(rng, &child_code);
                    }
                    if child_code.len() > MAX_CODE_LEN {
                        child_code.truncate(MAX_CODE_LEN);
                    }
                    let output = eval(&child_code).unwrap_or_default();
                    let fit = fitness(archive.contains_key(&output), child_code.len());
                    Individual {
                        code: child_code,
                        output,
                        fitness: fit,
                    }
                },
            )
            .collect();

        /* next generation */
        population.clear();
        population.extend(elites);
        population.extend(offspring);

        pb.inc(POP_SIZE as u64);
    }

    pb.finish_and_clear();
    drop(tx);
    writer.join().expect("writer thread panicked")?;
    println!("Done.");
    Ok(())
}

