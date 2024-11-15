use pulp::{Arch, Simd, WithSimd};
use reborrow::IntoConst;

struct Convolver<'a> {
    input: &'a [f32],
    kernel: &'a [f32],
    output: &'a mut [f32],
}

impl<'a> WithSimd for Convolver<'a> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {

        // assert that kernal length is less than 16
        let kernel_len = self.kernel.len();
        assert!(self.kernel.len() < 10);
        // assert that input length is greater than kernel length
        assert!(self.input.len() >= self.kernel.len());

        // assert that input is same length as output
        assert!(self.input.len() == self.output.len());

        // create a simd vector of the reciprocal of the kernel length
        let kernel_len_recip_simd = simd.splat_f32s(1.0 / kernel_len as f32);
        
        // make a vec to store each of the conv kernel elements as simd vectors
        let kernel_simd: Vec<_> = self.kernel.iter().map(|x| simd.splat_f32s(*x)).collect();

        // create a vec of shifted copies of the input
        // shape [shift,len(input)]
        let shifted_copies = create_shifted_copies(self.input, &kernel_len, &0.0);
        let (shifted_heads, _shifted_tails): (Vec<_>, Vec<_>) = shifted_copies
            .iter().map(|v| S::as_simd_f32s(&v)).unzip();

        let shifted_heads_t  = transpose_heads(simd, shifted_heads);
        // the tail is the remainder left after allocating v into simd vectors
        // len(tail) = len(v) % simd_vector_length
        let (out_head, out_tail) = S::as_mut_simd_f32s(self.output);

        for (i, column) in shifted_heads_t.into_iter().enumerate() {

            for (x, k) in column.iter().zip(kernel_simd.iter()) {
                out_head[i] = simd.mul_add_f32s(*x, *k, out_head[i]);
            }
            out_head[i] = simd.mul_f32s(out_head[i], kernel_len_recip_simd);
        }

        // now we need to handle the tails
    //     let tail_len = shifted_tails[0].len();
    //     if tail_len > 0 {

    //         for (x,k) in shifted_tails.iter().zip(kernel_simd.iter()) {
    //             out_tail = simd.mul_add_f32s(
    //                 simd.partial_load_f64s(x), 
    //                 *k, out_tail)
    //         }
    //         acc0 =
    //             simd.mul_add_f32s(simd.partial_load_f32s(x1), simd.partial_load_f32s(y1), acc0);
    //     }
    }
}
fn transpose_heads<S: Simd>(simd: S, heads: Vec<&[S::f32s]>) -> Vec<Vec<S::f32s>> {
    let mut transposed = vec![vec![simd.splat_f32s(0.0); heads.len()]; heads[0].len()];
    for i in 0..heads.len() {
        for j in 0..heads[0].len() {
            transposed[j][i] = heads[i][j];
        }
    }
    transposed
}

fn create_shifted_copies<'a, T: Clone>(v: &[T], num_shifts: &usize, pad_value: &T) -> Vec<Vec<T>> {

    let shifted_copies = (0..*num_shifts).map(|shift| {
        v.iter().skip(shift).chain(std::iter::repeat(pad_value).take(shift)).cloned().collect()
    }).collect::<Vec<_>>();


    // let mut shifted_copies = vec![vec![pad_value.clone(); v.len()]; num_shifts];
    // for shift in 0..num_shifts {
    //     for i in 0..v.len() {
    //         if i + shift < v.len() {
    //             shifted_copies[shift][i] = v[i+shift].clone();
    //         }
    //     }
    // }
    shifted_copies
}

fn naive_convolve(input: &[f32], kernel: &[f32]) -> Vec<f32> {
    let kernel_len = kernel.len();
    let kernel_len_f32 = kernel_len as f32;
    let output_len = input.len() - kernel_len + 1;
    let mut output = Vec::with_capacity(output_len);
    for i in 0..output_len {
        let mut acc = 0.0;
        for j in 0..kernel_len {
            acc += input[i + j] * kernel[j];
        }
        output.push(acc/ kernel_len_f32);
    }
    output
}
fn main() {
    let v = [1.0, 2.0, 3.0,4.0];// 4.0, 5.0,12.0,98.0,23.6,101.7];
    let k = [1.0, 2.0 ,3.0,7.0];
    // let outs = naive_convolve(&v, &k);
    // dbg!("{:?}", &outs);
    // let known_result = [14.0, 20.0, 26.0];
    // for (a, b) in outs.iter().zip(known_result.iter()) {
    //     assert!((a - b).abs() < 1e-6);
    // }
    let mut output = vec![0.0; v.len()];

    let convolver = Convolver {
        input: &v,
        kernel: &k,
        output: &mut output,
    };
    let arch = Arch::new();
    arch.dispatch(convolver);

    dbg!(&output);

    // do naive
    let outs = naive_convolve(&v, &k);
    dbg!("{:?}", &outs);

}
