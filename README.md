# k-Medoids Clustering in Rust with FasterPAM

For further details on the implemented algorithm FasterPAM, see:

> Erich Schubert, Peter J. Rousseeuw  
> **Fast and Eager k-Medoids Clustering:  
> O(k) Runtime Improvement of the PAM, CLARA, and CLARANS Algorithms**  
> Under review at Information Systems, Elsevier.  
> Preprint: <https://arxiv.org/abs/2008.05171>

an earlier (slower, and now obsolete) version was published as:

> Erich Schubert, Peter J. Rousseeuw:  
> **Faster k-Medoids Clustering: Improving the PAM, CLARA, and CLARANS Algorithms**  
> In: 12th International Conference on Similarity Search and Applications (SISAP 2019), 171-187.  
> <https://doi.org/10.1007/978-3-030-32047-8_16>  
> Preprint: <https://arxiv.org/abs/1810.05691>

This is a port of the original Java code from [ELKI](https://elki-project.github.io/) to Rust.

If you use this code in scientific work, please cite above papers. Thank you.

## Rust Dependencies

* [ndarray](https://docs.rs/ndarray/) for arrays
* [num-traits](https://docs.rs/num-traits/) for supporting different numeric types
* [rand](https://docs.rs/rand/) for random initialization

## License: GPL-3 or later

> This program is free software: you can redistribute it and/or modify
> it under the terms of the GNU General Public License as published by
> the Free Software Foundation, either version 3 of the License, or
> (at your option) any later version.
> 
> This program is distributed in the hope that it will be useful,
> but WITHOUT ANY WARRANTY; without even the implied warranty of
> MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
> GNU General Public License for more details.
> 
> You should have received a copy of the GNU General Public License
> along with this program.  If not, see <https://www.gnu.org/licenses/>.

## FAQ: Why GPL and not Apache/MIT/BSD?

Because copyleft software such as Linux got us where we are now.

Tit for tat: you get to use my code, I get to use your code.
