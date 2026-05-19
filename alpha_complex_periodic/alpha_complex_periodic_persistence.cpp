#include <iostream>
#include <string>
#include <vector>
#include <limits>
#include <random>
#include <cmath>
#include <cstdlib>

#include <gudhi/Alpha_complex_3d.h>
#include <gudhi/Simplex_tree.h>
#include <gudhi/Persistent_cohomology.h>


using Simplex_tree = Gudhi::Simplex_tree<Gudhi::Simplex_tree_options_fast_persistence>;
using Filtration_value = Simplex_tree::Filtration_value;
using Persistent_cohomology =
    Gudhi::persistent_cohomology::Persistent_cohomology<Simplex_tree, Gudhi::persistent_cohomology::Field_Zp>;


// convert 2D vector of 3D positions to vector of point objects
// input: [[x_i, y_i, z_i], ...]
// output: [point_i, ...]
template <typename AlphaComplex3d>
std::vector<typename AlphaComplex3d::Point_3> coords_to_points(const std::vector<std::vector<double>> &coords) {
    using Point = typename AlphaComplex3d::Point_3;

    std::vector<Point> points(coords.size());
    for (std::size_t i = 0; i < coords.size(); i++) {
        points[i] = Point(coords[i][0], coords[i][1], coords[i][2]);
    }

    return points;
}


// returns 3D vector of persistence pairs
// [dim_0, dim_1, dim_2]
//   dim_i: [[birth_j, death_j], ...]
std::vector<std::vector<std::vector<double>>> calc_persistence(const std::vector<std::vector<double>> &coords, int coeff_field, double min_persistence, bool fast, bool exact, double boxsize = 0.) {

    Gudhi::alpha_complex::complexity complexity = Gudhi::alpha_complex::complexity::SAFE;
    if (exact) {
        complexity = Gudhi::alpha_complex::complexity::EXACT;
    } else if (fast) {
        complexity = Gudhi::alpha_complex::complexity::FAST;
    }

    Simplex_tree simplex_tree;

    // Alpha complex can be periodic/non-periodic and safe/fast/exact
    // complexity affects computation of filtration values
    if (boxsize > 0) {
        switch (complexity) {
            case Gudhi::alpha_complex::complexity::FAST: {
                using Alpha_complex_3d = Gudhi::alpha_complex::Alpha_complex_3d<Gudhi::alpha_complex::complexity::FAST, false, true>;
                auto points = coords_to_points<Alpha_complex_3d>(coords);
                Alpha_complex_3d alpha_complex(points, 0, 0, 0, boxsize, boxsize, boxsize);
                alpha_complex.create_complex(simplex_tree);
                break;
            }
            case Gudhi::alpha_complex::complexity::SAFE: {
                using Alpha_complex_3d = Gudhi::alpha_complex::Alpha_complex_3d<Gudhi::alpha_complex::complexity::SAFE, false, true>;
                auto points = coords_to_points<Alpha_complex_3d>(coords);
                Alpha_complex_3d alpha_complex(points, 0, 0, 0, boxsize, boxsize, boxsize);
                alpha_complex.create_complex(simplex_tree);
                break;
            }
            case Gudhi::alpha_complex::complexity::EXACT: {
                using Alpha_complex_3d = Gudhi::alpha_complex::Alpha_complex_3d<Gudhi::alpha_complex::complexity::EXACT, false, true>;
                auto points = coords_to_points<Alpha_complex_3d>(coords);
                Alpha_complex_3d alpha_complex(points, 0, 0, 0, boxsize, boxsize, boxsize);
                alpha_complex.create_complex(simplex_tree);
                break;
            }
        }
    } else {
        switch (complexity) {
            case Gudhi::alpha_complex::complexity::FAST: {
                using Alpha_complex_3d = Gudhi::alpha_complex::Alpha_complex_3d<Gudhi::alpha_complex::complexity::FAST, false, false>;
                auto points = coords_to_points<Alpha_complex_3d>(coords);
                Alpha_complex_3d alpha_complex(points);
                alpha_complex.create_complex(simplex_tree);
                break;
            }
            case Gudhi::alpha_complex::complexity::SAFE: {
                using Alpha_complex_3d = Gudhi::alpha_complex::Alpha_complex_3d<Gudhi::alpha_complex::complexity::SAFE, false, false>;
                auto points = coords_to_points<Alpha_complex_3d>(coords);
                Alpha_complex_3d alpha_complex(points);
                alpha_complex.create_complex(simplex_tree);
                break;
            }
            case Gudhi::alpha_complex::complexity::EXACT: {
                using Alpha_complex_3d = Gudhi::alpha_complex::Alpha_complex_3d<Gudhi::alpha_complex::complexity::EXACT, false, false>;
                auto points = coords_to_points<Alpha_complex_3d>(coords);
                Alpha_complex_3d alpha_complex(points);
                alpha_complex.create_complex(simplex_tree);
                break;
            }
        }
    }

    int st_dimension = simplex_tree.dimension();

    // calculate persistent homology
    Persistent_cohomology pcoh(simplex_tree, true);
    pcoh.init_coefficients(coeff_field);
    pcoh.compute_persistent_cohomology(min_persistence);

    std::vector<std::vector<std::vector<double>>> persistence_pairs(st_dimension);
    std::vector<std::pair<Filtration_value, Filtration_value>> pairs_d;
    for (int dim = 0; dim < st_dimension; dim++) {
        pairs_d = pcoh.intervals_in_dimension(dim);
        std::vector<std::vector<double>> pairs_d_vec(pairs_d.size());
        for (std::size_t i = 0; i < pairs_d.size(); i++) {
            // take sqrt to get alpha instead of the filtration value (alpha^2)
            pairs_d_vec[i] = {sqrt(pairs_d[i].first), sqrt(pairs_d[i].second)};
        }
        persistence_pairs[dim] = pairs_d_vec;
    }

    return persistence_pairs;
}


int main() {
    // For some reason need ~ >50 points for periodic triangulation to work
    int N_points = 15000;
    double boxsize = 5;

    std::default_random_engine generator;
    generator.seed(10);
    std::uniform_real_distribution<double> distribution(0, boxsize);

    // generate random points inside box
    std::vector<std::vector<double>> points;
    for (int i = 0; i < N_points; i++) {
        std::vector<double> point = {distribution(generator), distribution(generator), distribution(generator)};
        points.push_back(point);
    }

    std::cout << "Points:" << std::endl;
    for (int i = 0; i < N_points; i++) {
        std::cout << "\t" << points[i][0] << ", " << points[i][1] << ", " << points[i][2] << std::endl;
    }

    std::vector<std::vector<std::vector<double>>> persistence_pairs = calc_persistence(points, 2, 0., false, false, boxsize);

        for (std::size_t dim = 0; dim < persistence_pairs.size(); dim++) {
        std::cout << "H" << dim << std::endl;
        for (std::size_t i = 0; i < persistence_pairs[dim].size(); i++) {
            std::cout << "\t" << persistence_pairs[dim][i][0] << "  " << persistence_pairs[dim][i][1] << std::endl;
        }
    }
}
