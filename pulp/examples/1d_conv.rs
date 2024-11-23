use pulp::{Arch, Simd, WithSimd};
use diol::prelude::*;
use aligned_vec::avec;
const KERNEL_RECIP_3: f32 = 1.0 / 3.0;

struct Convolver3k<'a> {
    input: &'a [f32],
    kernel: &'a [f32; 3],
    output: &'a mut [f32],
}

struct TransposedHeadsIter<'a, S: Simd> {
    heads: [&'a [S::f32s]; 3],
    current_idx: usize,
}

impl<'a, S: Simd> Iterator for TransposedHeadsIter<'a, S> {
    type Item = [S::f32s; 3];

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.heads[0].len() {
            return None;
        }
        let result = [
            self.heads[0][self.current_idx],
            self.heads[1][self.current_idx],
            self.heads[2][self.current_idx],
        ];
        self.current_idx += 1;
        Some(result)
    }
}

impl<'a> WithSimd for Convolver3k<'a> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        // assert that kernal length is less than 16
        assert_eq!(self.kernel.len(), 3);
        assert!(self.kernel.len() < 4);
        // assert that input is same length as output
        assert!(self.input.len() == self.output.len());
        // create a simd vector of the reciprocal of the kernel length
        let kernel_len_recip_simd = simd.splat_f32s(KERNEL_RECIP_3);
        let kernel_simd: [S::f32s; 3] = [
            simd.splat_f32s(self.kernel[0]),
            simd.splat_f32s(self.kernel[1]),
            simd.splat_f32s(self.kernel[2]),
        ];
        // create a vec of shifted copies of the input
        // shape [shift,len(input)]
        let shifted_copies = create_3_shifted_copies(self.input);
        let shifted_splits = [
            S::as_simd_f32s(shifted_copies[0]),
            S::as_simd_f32s(shifted_copies[1]),
            S::as_simd_f32s(shifted_copies[2]),
        ];

        let shifted_heads = [
            shifted_splits[0].0,
            shifted_splits[1].0,
            shifted_splits[2].0,
        ];

        let shifted_tails = [
            shifted_splits[0].1,
            shifted_splits[1].1,
            shifted_splits[2].1,
        ];

        let shifted_heads_iterator = transpose_heads::<S>(shifted_heads);
        // the tail is the remainder left after allocating v into simd vectors
        // len(tail) = len(v) % simd_vector_length
        let (out_head, out_tail) = S::as_mut_simd_f32s(self.output);

        for (i, column) in shifted_heads_iterator.enumerate() {
            out_head[i] = simd.mul_add_f32s(column[0], kernel_simd[0], out_head[i]);
            out_head[i] = simd.mul_add_f32s(column[1], kernel_simd[1], out_head[i]);
            out_head[i] = simd.mul_add_f32s(column[2], kernel_simd[2], out_head[i]);
            out_head[i] = simd.mul_f32s(out_head[i], kernel_len_recip_simd);
        }

        // now we need to handle the tails
        // can i naive iterate on the tails or is that too slow?

        if out_tail.len() > 0 {
            for i in 0..out_tail.len() - 2 {
                out_tail[i] = shifted_tails[0][i] * self.kernel[0]
                    + shifted_tails[1][i] * self.kernel[1]
                    + shifted_tails[2][i] * self.kernel[2];
                out_tail[i] *= KERNEL_RECIP_3;
            }
        }
    }
}

fn transpose_heads<S: Simd>(heads: [&[S::f32s]; 3]) -> TransposedHeadsIter<S> {
    TransposedHeadsIter {
        heads,
        current_idx: 0,
    }
}

fn create_3_shifted_copies<'a, T: Clone>(v: &'a [T]) -> [&'a [T]; 3] {
    [&v[0..v.len() - 2], &v[1..v.len() - 1], &v[2..v.len()]]
}

fn naive_convolve_3k(input: &[f32], kernel: &[f32], output: &mut [f32]) {
    assert_eq!(kernel.len(), 3);
    let output_len = output.len();
    for (i, out_element) in output.iter_mut().take(output_len - 2).enumerate() {
        *out_element =
            (input[i] * kernel[0] + input[i + 1] * kernel[1] + input[i + 2] * kernel[2]) / 3.0;
    }
    *output.last_mut().unwrap() = 0.0;
    return;
}

fn bench_naive_convolve_3k(
    bencher: Bencher,
    PlotArg(n): PlotArg
) {
    let v = &*vec![1.0; n];
    let k = &[1.0, 2.0, 3.0];
    let output = &mut *vec![0.0; n];

    bencher.bench(|| naive_convolve_3k(v, k, output));
}

fn bench_simd_3k(
    bencher: Bencher,
    PlotArg(n): PlotArg
) {
    let v = &*avec![1.0; n];
    let k = &[1.0, 2.0, 3.0];
    let output = &mut *avec![0.0; n];

    
    let arch = Arch::new();
    bencher.bench(|| arch.dispatch(Convolver3k {
        input: v,
        kernel: k,
        output,
    }));
}

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);

    let mut params = vec![];
    for n in 5..10 {
        params.push(10 * n);
    }

    for n in 5..10 {
        params.push(100 * n);
    }
    for n in 5..10 {
        params.push(1000 * n);
    }

    bench.register_many(list![
        bench_naive_convolve_3k,
        bench_simd_3k,
    ], params.iter().copied().map(PlotArg));

    bench.run()?;
    Ok(())

}
