use crate::arrayadapter::ArrayAdapter;
use crate::arrayadapter3d::ArrayAdapter3d;
use crate::util::*;
use core::ops::AddAssign;
use num_traits::{Signed, Zero};
use std::convert::From;
use ndarray::Array3;
use ndarray::Array2;
use ndarray::Array1;
use ndarray::Axis;
use std::ops::Range;

/// Run the FasterPAM algorithm.
///
/// If used multiple times, it is better to additionally shuffle the input data,
/// to increase randomness of the solutions found and hence increase the chance
/// of finding a better solution.
///
/// * type `M` - matrix data type such as `ndarray::Array2` or `kmedoids::arrayadapter::LowerTriangle`
/// * type `N` - number data type such as `u32` or `f64`
/// * type `L` - number data type such as `i64` or `f64` for the loss (must be signed)
/// * `mat` - a pairwise distance matrix
/// * `med` - the list of medoids
/// * `maxiter` - the maximum number of iterations allowed
///
/// returns a tuple containing:
/// * the final loss
/// * the final cluster assignment
/// * the number of iterations needed
/// * the number of swaps performed
///
/// ## Panics
///
/// * panics when the dissimilarity matrix is not square
/// * panics when k is 0 or larger than N
///
/// ## Example
/// Given a dissimilarity matrix of size 4 x 4, use:
/// ```
/// let data = ndarray::arr2(&[[0,1,2,3],[1,0,4,5],[2,4,0,6],[3,5,6,0]]);
/// let mut meds = kmedoids::random_initialization(4, 2, &mut rand::thread_rng());
/// let (loss, assi, n_iter, n_swap): (f64, _, _, _) = kmedoids::fasterpam(&data, &mut meds, 100);
/// println!("Loss is: {}", loss);
/// ```
pub fn fasterpam<M, N, L>(
	mat: &M,
	med: &mut Vec<usize>,
	maxiter: usize,
) -> (L, Vec<usize>, usize, usize)
where
	N: Zero + PartialOrd + Copy,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N>,
	M: ArrayAdapter<N>,
{
	let (n, k) = (mat.len(), med.len());
	if k == 1 {
		let assi = vec![0; n];
		let (swapped, loss) = choose_medoid_within_partition::<M, N, L>(mat, &assi, med, 0);
		return (loss, assi, 1, if swapped { 1 } else { 0 });
	}
	let (mut loss, mut data) = initial_assignment(mat, med);
	debug_assert_assignment(mat, med, &data);
	let mut removal_loss = vec![L::zero(); k];
	update_removal_loss(&data, &mut removal_loss);
	let (mut lastswap, mut n_swaps, mut iter) = (n, 0, 0);
	while iter < maxiter {
		iter += 1;
		let (swaps_before, lastloss) = (n_swaps, loss);
		for j in 0..n {
			if j == lastswap {
				break;
			}
			if j == med[data[j].near.i as usize] {
				continue; // This already is a medoid
			}
			let (change, b) = find_best_swap(mat, &removal_loss, &data, j);
			if change >= L::zero() {
				continue; // No improvement
			}
			n_swaps += 1;
			lastswap = j;
			// perform the swap
			loss = do_swap(mat, med, &mut data, b, j);
			update_removal_loss(&data, &mut removal_loss);
		}
		if n_swaps == swaps_before || loss >= lastloss {
			break; // converged
		}
	}
	let assi = data.iter().map(|x| x.near.i as usize).collect();
	(loss, assi, iter, n_swaps)
}

/// Run the FasterPAM algorithm with additional randomization.
///
/// This increases the chance of finding a better solution when used multiple times,
/// as it decreases the dependency on the input data order.
///
/// * type `M` - matrix data type such as `ndarray::Array2` or `kmedoids::arrayadapter::LowerTriangle`
/// * type `N` - number data type such as `u32` or `f64`
/// * type `L` - number data type such as `i64` or `f64` for the loss (must be signed)
/// * `mat` - a pairwise distance matrix
/// * `med` - the list of medoids
/// * `maxiter` - the maximum number of iterations allowed
/// * `rng` - random number generator for shuffling the input data
///
/// returns a tuple containing:
/// * the final loss
/// * the final cluster assignment
/// * the number of iterations needed
/// * the number of swaps performed
///
/// ## Panics
///
/// * panics when the dissimilarity matrix is not square
/// * panics when k is 0 or larger than N
///
/// ## Example
/// Given a dissimilarity matrix of size 4 x 4, use:
/// ```
/// let data = ndarray::arr2(&[[0,1,2,3],[1,0,4,5],[2,4,0,6],[3,5,6,0]]);
/// let mut meds = kmedoids::random_initialization(4, 2, &mut rand::thread_rng());
/// let (loss, assi, n_iter, n_swap): (f64, _, _, _) = kmedoids::rand_fasterpam(&data, &mut meds, 100, &mut rand::thread_rng());
/// println!("Loss is: {}", loss);
/// ```
#[cfg(feature = "rand")]
pub fn rand_fasterpam<M, N, L>(
	mat: &M,
	med: &mut Vec<usize>,
	maxiter: usize,
	rng: &mut impl rand::Rng,
) -> (L, Vec<usize>, usize, usize)
where
	N: Zero + PartialOrd + Copy,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N>,
	M: ArrayAdapter<N>,
{
	let (n, k) = (mat.len(), med.len());
	if k == 1 {
		let assi = vec![0; n];
		let (swapped, loss) = choose_medoid_within_partition::<M, N, L>(mat, &assi, med, 0);
		return (loss, assi, 1, if swapped { 1 } else { 0 });
	}
	let (mut loss, mut data) = initial_assignment(mat, med);
	debug_assert_assignment(mat, med, &data);

	let mut removal_loss = vec![L::zero(); k];
	update_removal_loss(&data, &mut removal_loss);
	let (mut lastswap, mut n_swaps, mut iter) = (n, 0, 0);
	let seq = rand::seq::index::sample(rng, n, n); // random shuffling
	while iter < maxiter {
		iter += 1;
		let (swaps_before, lastloss) = (n_swaps, loss);
		for j in seq.iter() {
			if j == lastswap {
				break;
			}
			if j == med[data[j].near.i as usize] {
				continue; // This already is a medoid
			}
			let (change, b) = find_best_swap(mat, &removal_loss, &data, j);
			if change >= L::zero() {
				continue; // No improvement
			}
			n_swaps += 1;
			lastswap = j;
			// perform the swap
			loss = do_swap(mat, med, &mut data, b, j);
			update_removal_loss(&data, &mut removal_loss);
		}
		if n_swaps == swaps_before || loss >= lastloss {
			break; // converged
		}
	}
	let assi = data.iter().map(|x| x.near.i as usize).collect();
	(loss, assi, iter, n_swaps)
}

#[cfg(feature = "rand")]
pub fn fasterpam_time<M,N,L>(
	mat: &M,
	medioids: &mut Vec<usize>,
	maxiter: usize,
	drift_time_window: usize,
	max_drift: usize,
	rng: &mut impl rand::Rng,
	online: bool,
	) -> (L, Array2<usize>, usize, usize, Array2<usize>)
where
N: Zero + PartialOrd + Copy + From<f64>,
L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N>,
M: ArrayAdapter3d<N>,
{
	// Get data dimensions
	let t: usize = mat.get_dim_size(0);
	let n: usize = mat.get_dim_size(1);
	let k: usize = medioids.len();

	let mut total_assignments = Array2::<usize>::zeros((t, n));
	let mut total_medioids = Array2::<usize>::zeros((t, k));
	
	let mut start: usize = 1;
	if !online{
		// Get first time slice
		let distance_init = mat.single_slice(0);
		let (_loss, assi, _iter, _n_swaps): (L, Vec<usize>, usize, usize)  = rand_fasterpam(&distance_init, medioids, maxiter, rng);
		
		let init_med_array = Array1::from(medioids.clone().into_boxed_slice().into_vec());
		let init_assi_array = Array1::from(assi);

		total_medioids.row_mut(0).assign(&init_med_array);
		total_assignments.row_mut(0).assign(&init_assi_array);

		}
	else{
		start = 0;
	}

	for time in start..t{
		let curr_distance = mat.single_slice(time);
		let mut removal_loss = vec![L::zero(); k];
		let (mut _loss, mut assignment) : (L, Vec<Rec<N>>) = initial_assignment(&curr_distance, medioids);

		update_removal_loss(&assignment, &mut removal_loss);

		let curr_mediods = medioids.clone();
		
		//randomly shuffle the medioids
		let med_seq = rand::seq::index::sample(rng, k, k); // random shuffling
		
		//for (medioid_index, medioid) in curr_mediods.iter().enumerate(){
		for medioid_index in med_seq.iter(){
			let medioid = curr_mediods[medioid_index];

			let medioid_connections = find_center_connections::<M,N,L>(
				medioid,
				mat, 
				time, 
				drift_time_window,
				max_drift,
			);

			//randomly shuffle the medioid connections
			let connections_len = medioid_connections.len();
			let connections_seq = rand::seq::index::sample(rng, connections_len, connections_len);

			// for connection in mediod_connections.iter(){
			for connection_index in connections_seq.iter(){
				let connection = medioid_connections[connection_index];
				if connection == medioids[assignment[connection].near.i as usize] {
					continue; // This already is a medoid
			}
				let (change, b) = find_swap_for_medioid(
					&curr_distance,
					&mut removal_loss,
					&assignment,
					connection, 
					medioid_index,
				);

				if change >= L::zero() {
					continue; // No improvement
				}
				// perform the swap
				let _loss: L = do_swap(&curr_distance, medioids, &mut assignment, b, connection);
				update_removal_loss(&assignment, &mut removal_loss);
				}
			}
		let assi: Vec<usize> = assignment.iter().map(|x| x.near.i as usize).collect();
		
		let init_med_array = Array1::from(medioids.clone().into_boxed_slice().into_vec());
		let init_assi_array = Array1::from(assi);
		total_medioids.row_mut(time).assign(&init_med_array);
		total_assignments.row_mut(time).assign(&init_assi_array);

	}
	// Store assi in some 3D struct
	(L::zero(), total_assignments, 0, 0, total_medioids)
}

// Find center connections
#[inline]
pub(crate) fn find_center_connections<M,N,L> (
    current_center: usize,
    distance_matrix: &M,
    time: usize,
    drift_time_window: usize,
    max_drift: usize,
) -> Vec<usize>
where 
N: Zero + PartialOrd + Copy + From<f64>,
M: ArrayAdapter3d<N>,
L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N>,

{
    let num_points = distance_matrix.get_dim_size(1);
	
	// Distances matrices for "center connectivity" previous time steps
	let slice_range: Range<usize>;

	if time<drift_time_window{
		slice_range = 0..time;
	}
	else{
		slice_range = (time - drift_time_window)..time;
	}

    let drift_time_slices: Array3<N> =
        distance_matrix.multi_slice(slice_range);
	
	let target_sum: f64 = max_drift as f64 * f64::min(time as f64 , drift_time_window as f64);

	// let target_sum: u32 = max_drift * min(time , drift_time_window);

	let sum_distances = drift_time_slices.index_axis(Axis(1), current_center).sum_axis(Axis(0));
	let connected_vertices: Vec<usize> = (0..num_points).filter(|&i| sum_distances[i] <= N::from(target_sum)).collect();
	connected_vertices  

    // center_connections
}

/// Perform the initial assignment to medoids
#[inline]
pub(crate) fn initial_assignment<M, N, L>(mat: &M, med: &[usize]) -> (L, Vec<Rec<N>>)
where
	N: Zero + PartialOrd + Copy,
	L: AddAssign + Zero + PartialOrd + Copy + From<N>,
	M: ArrayAdapter<N>,
{
	let (n, k) = (mat.len(), med.len());
	assert!(mat.is_square(), "Dissimilarity matrix is not square");
	assert!(n <= u32::MAX as usize, "N is too large");
	assert!(k > 0 && k < u32::MAX as usize, "invalid N");
	assert!(k <= n, "k must be at most N");
	let mut data = vec![Rec::<N>::empty(); mat.len()];

	let firstcenter = med[0];
	let loss = data
		.iter_mut()
		.enumerate()
		.map(|(i, cur)| {
			*cur = Rec::new(0, mat.get(i, firstcenter), u32::MAX, N::zero());
			for (m, &me) in med.iter().enumerate().skip(1) {
				let d = mat.get(i, me);
				if d < cur.near.d || i == me {
					cur.seco = cur.near;
					cur.near = DistancePair { i: m as u32, d };
				} else if cur.seco.i == u32::MAX || d < cur.seco.d {
					cur.seco = DistancePair { i: m as u32, d };
				}
			}
			L::from(cur.near.d)
		})
		.reduce(L::add)
		.unwrap();
	(loss, data)
}

/// Find the best swap for object j - FastPAM version
#[inline]
pub(crate) fn find_swap_for_medioid<M, N, L>(
	mat: &M,
	removal_loss: &[L],
	data: &[Rec<N>],
	j: usize,
	med_to_replace_index: usize,
) -> (L, usize)
where
	N: Zero + PartialOrd + Copy,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N>,
	M: ArrayAdapter<N>,
{
	let mut ploss = removal_loss.to_vec();

	
	// Improvement from the journal version:
	let mut acc = L::zero();
	for (o, reco) in data.iter().enumerate() {
		let doj = mat.get(o, j);
		// New medoid is closest:
		if doj < reco.near.d {
			acc += L::from(doj) - L::from(reco.near.d);
			// loss already includes ds - dn, remove
			ploss[reco.near.i as usize] += L::from(reco.near.d) - L::from(reco.seco.d);
		} else if doj < reco.seco.d {
			// loss already includes ds - dn, adjust to d(xo) - dn
			ploss[reco.near.i as usize] += L::from(doj) - L::from(reco.seco.d);
		}
	}
	// let (b, bloss) = find_min(&mut ploss.iter());
	// Find index and value for med_to_replace 
	let med_loss = ploss[med_to_replace_index];
	(med_loss + acc, med_to_replace_index) // add the shared accumulator
}

/// Find the best swap for object j - FastPAM version
#[inline]
pub(crate) fn find_best_swap<M, N, L>(
	mat: &M,
	removal_loss: &[L],
	data: &[Rec<N>],
	j: usize,
) -> (L, usize)
where
	N: Zero + PartialOrd + Copy,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N>,
	M: ArrayAdapter<N>,
{
	let mut ploss = removal_loss.to_vec();
	// Improvement from the journal version:
	let mut acc = L::zero();
	for (o, reco) in data.iter().enumerate() {
		let doj = mat.get(o, j);
		// New medoid is closest:
		if doj < reco.near.d {
			acc += L::from(doj) - L::from(reco.near.d);
			// loss already includes ds - dn, remove
			ploss[reco.near.i as usize] += L::from(reco.near.d) - L::from(reco.seco.d);
		} else if doj < reco.seco.d {
			// loss already includes ds - dn, adjust to d(xo) - dn
			ploss[reco.near.i as usize] += L::from(doj) - L::from(reco.seco.d);
		}
	}
	let (b, bloss) = find_min(&mut ploss.iter());
	(bloss + acc, b) // add the shared accumulator
}

/// Update the loss when removing each medoid
pub(crate) fn update_removal_loss<N, L>(data: &[Rec<N>], loss: &mut Vec<L>)
where
	N: Zero + Copy,
	L: AddAssign + Signed + Copy + Zero + From<N>,
{
	loss.fill(L::zero()); // stable since 1.50
	for rec in data.iter() {
		loss[rec.near.i as usize] += L::from(rec.seco.d) - L::from(rec.near.d);
		// as N might be unsigned
	}
}

/// Update the second nearest medoid information
/// Called after each swap.
#[inline]
pub(crate) fn update_second_nearest<M, N>(
	mat: &M,
	med: &[usize],
	n: usize,
	b: usize,
	o: usize,
	doj: N,
) -> DistancePair<N>
where
	N: PartialOrd + Copy,
	M: ArrayAdapter<N>,
{
	let mut s = DistancePair::new(b as u32, doj);
	for (i, &mi) in med.iter().enumerate() {
		if i == n || i == b {
			continue;
		}
		let d = mat.get(o, mi);
		if d < s.d {
			s = DistancePair::new(i as u32, d);
		}
	}
	s
}

/// Perform a single swap
#[inline]
pub(crate) fn do_swap<M, N, L>(
	mat: &M,
	med: &mut Vec<usize>,
	data: &mut Vec<Rec<N>>,
	b: usize,
	j: usize,
) -> L
where
	N: Zero + PartialOrd + Copy,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N>,
	M: ArrayAdapter<N>,
{
	let n = mat.len();
	assert!(b < med.len(), "invalid medoid number");
	assert!(j < n, "invalid object number");
	med[b] = j;
	data.iter_mut()
		.enumerate()
		.map(|(o, reco)| {
			if o == j {
				if reco.near.i != b as u32 {
					reco.seco = reco.near;
				}
				reco.near = DistancePair::new(b as u32, N::zero());
				return L::zero();
			}
			let doj = mat.get(o, j);
			// Nearest medoid is gone:
			if reco.near.i == b as u32 {
				if doj < reco.seco.d {
					reco.near = DistancePair::new(b as u32, doj);
				} else {
					reco.near = reco.seco;
					reco.seco = update_second_nearest(mat, med, reco.near.i as usize, b, o, doj);
				}
			} else {
				// nearest not removed
				if doj < reco.near.d {
					reco.seco = reco.near;
					reco.near = DistancePair::new(b as u32, doj);
				} else if doj < reco.seco.d {
					reco.seco = DistancePair::new(b as u32, doj);
				} else if reco.seco.i == b as u32 {
					// second nearest was replaced
					reco.seco = update_second_nearest(mat, med, reco.near.i as usize, b, o, doj);
				}
			}
			L::from(reco.near.d)
		})
		.reduce(L::add)
		.unwrap()
}

#[cfg(test)]
mod tests {
	// TODO: use a larger, much more interesting example.
	use crate::{arrayadapter::LowerTriangle, fasterpam, silhouette, util::assert_array};

	#[test]
	fn testfasterpam_simple() {
		let data = LowerTriangle {
			n: 5,
			data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
		};
		let mut meds = vec![0, 1];
		let (loss, assi, n_iter, n_swap): (i64, _, _, _) = fasterpam(&data, &mut meds, 10);
		let (sil, _): (f64, _) = silhouette(&data, &assi, false);
		assert_eq!(loss, 4, "loss not as expected");
		assert_eq!(n_swap, 2, "swaps not as expected");
		assert_eq!(n_iter, 2, "iterations not as expected");
		assert_array(assi, vec![0, 0, 0, 1, 1], "assignment not as expected");
		assert_array(meds, vec![0, 3], "medoids not as expected");
		assert_eq!(sil, 0.7522494172494172, "Silhouette not as expected");
	}

	#[test]
	fn testfasterpam_single_cluster() {
		let data = LowerTriangle {
			n: 5,
			data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
		};
		let mut meds = vec![1]; // So we need one swap
		let (loss, assi, n_iter, n_swap): (i64, _, _, _) = fasterpam(&data, &mut meds, 10);
		let (sil, _): (f64, _) = silhouette(&data, &assi, false);
		assert_eq!(loss, 14, "loss not as expected");
		assert_eq!(n_swap, 1, "swaps not as expected");
		assert_eq!(n_iter, 1, "iterations not as expected");
		assert_array(assi, vec![0, 0, 0, 0, 0], "assignment not as expected");
		assert_array(meds, vec![0], "medoids not as expected");
		assert_eq!(sil, 0., "Silhouette not as expected");
	}

	#[cfg(feature = "rand")]
	use crate::rand_fasterpam;
	#[cfg(feature = "rand")]
	use rand::{rngs::StdRng, SeedableRng};
	#[cfg(feature = "rand")]
	#[test]
	fn testrand_fasterpam() {
		let data = LowerTriangle {
			n: 5,
			data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
		};
		let mut meds = vec![0, 1];
		let mut rng = StdRng::seed_from_u64(1);
		let (loss, assi, n_iter, n_swap): (i64, _, _, _) =
			rand_fasterpam(&data, &mut meds, 10, &mut rng);
		let (sil, _): (f64, _) = silhouette(&data, &assi, false);
		assert_eq!(loss, 4, "loss not as expected");
		assert_eq!(n_swap, 1, "swaps not as expected");
		assert_eq!(n_iter, 2, "iterations not as expected");
		assert_array(assi, vec![0, 0, 0, 1, 1], "assignment not as expected");
		assert_array(meds, vec![0, 4], "medoids not as expected");
		assert_eq!(sil, 0.7522494172494172, "Silhouette not as expected");
	}

	#[cfg(feature = "rand")]
	// use crate::fasterpam_time;
	#[test]
	fn test_center_connections() {
    use crate::find_center_connections;

		let distance_matrix: ndarray::prelude::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 3]>> = ndarray::Array3::from_shape_vec((4, 4, 4), vec![
        0.0, 3.0, 1.0, 2.0, 3.0, 0.0, 2.0, 3.0,
        1.0, 2.0, 0.0, 3.0, 2.0, 3.0, 3.0, 0.0,
        0.0, 3.0, 1.0, 2.0, 3.0, 0.0, 2.0, 3.0,
        1.0, 2.0, 0.0, 3.0, 2.0, 3.0, 3.0, 0.0,
        0.0, 3.0, 1.0, 2.0, 3.0, 0.0, 2.0, 3.0,
        1.0, 2.0, 0.0, 3.0, 2.0, 3.0, 3.0, 0.0,
        0.0, 3.0, 1.0, 2.0, 3.0, 0.0, 2.0, 3.0,
        1.0, 2.0, 0.0, 3.0, 2.0, 3.0, 3.0, 0.0,]).unwrap();
		
		let connections: Vec<usize> = find_center_connections::<ndarray::prelude::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 3]>>, f64, f64>(2, &distance_matrix, 2, 2, 2);
		assert_array(connections, vec![0,1,2], "connections not as expected");
		
		let connections: Vec<usize> = find_center_connections::<ndarray::prelude::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 3]>>, f64, f64>( 3, &distance_matrix, 2, 2, 2);
		assert_array(connections, vec![0,3], "connections not as expected");
	}
}
