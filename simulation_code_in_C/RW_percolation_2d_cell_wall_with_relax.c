#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h> // Include for boolean data type
#include "SFMT.h"
#include <time.h> // for running time checks
#include <stdlib.h>  // Include the stdlib.h header for random number generation
#include <omp.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <sys/types.h>

#define NUM_BATCHES 160 //total number of batches / obstacle realizations to run
#define NUM_traj 100 //total number of trajectories for each batch
#define N_STEPS_LOG10 8.7 //total number of steps for each trajectory in log10
#define STEP_GENERATION_BATCH_SIZE 1000000 //increments of steps to generate at once every sequence until crossing N_STEPS
#define W_lattice 8 //the width of the cell wall in units of mesh_res
#define CELL_HALF_WIDTH 100 // in mesh_res units, used to determin when do we bump against the wall on the other side of the cell
#define h_lattice 1001 //the height of the cell wall in units of mesh_res
#define mesh_res 4 // this is ~4nm
#define save_path "cell_wall_relax_runs/Fig_3B/10_percent"

struct step{
	int x;
	int y;
};

struct step make_step(double rand_value, double F){	
	struct step step0;
	double norm = 1.0 / (2.0 + exp(F/2) + exp(-F/2));
	double p_right = norm * exp(F/2);
	double p_left = norm * exp(-F/2);
	double p_up = norm;
	double p_down = norm;
	if (rand_value < p_right) {
	  // Right
		step0.x = 1;
		step0.y = 0;
	} else if (rand_value < p_right + p_left) {
	  // left
		step0.x = -1;
		step0.y = 0;
	} else if (rand_value < p_right + p_left + p_up) {
	  // up
		step0.x = 0;
		step0.y = 1;
	} else {
	  // down
		step0.x = 0;
		step0.y = -1;
	}
	
	return step0;
}

struct BitField{
    unsigned int flag : 1; // Define a 1-bit bit-field variable named 'flag'
};

// Custom choice function
void generate_randvec(sfmt_t* sfmt, double randvec[STEP_GENERATION_BATCH_SIZE]) {
	// sfmt is already initialized and we only draw the random numbers
	// for example : double rand_value = sfmt_genrand_real1(rng); rng here is sfmt
	
	const int NUM = STEP_GENERATION_BATCH_SIZE;
    int size;
    
    size = sfmt_get_min_array_size64(sfmt);
    if (size < NUM) {
		size = NUM;
    }
	if (size % 2 != 0) {
		size++;  // Make sure it's even
	}
	
    uint64_t *array;

	#if defined(__APPLE__) || \
		(defined(__FreeBSD__) && __FreeBSD__ >= 3 && __FreeBSD__ <= 6)
		//printf("malloc used\n");
		array = malloc(sizeof(double) * size);
		if (array == NULL) {
		printf("can't allocate memory.\n");
		}
	#elif defined(_POSIX_C_SOURCE)
		//printf("posix_memalign used\n");
		if (posix_memalign((void **)&array, 16, sizeof(double) * size) != 0) {
		printf("can't allocate memory.\n");
		}
	//#elif defined(__GNUC__) && (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 3))
		//printf("memalign used\n");
		//array = memalign(16, sizeof(double) * size);
		//if (array == NULL) {
		//printf("can't allocate memory.\n");
		//return 1;
		//}
	#else /* in this case, gcc doesn't support SSE2 */
		//printf("malloc used\n");
		array = malloc(sizeof(double) * size);
		if (array == NULL) {
		printf("can't allocate memory.\n");
		}
	#endif
	
    sfmt_fill_array64(sfmt, array, size);
    
    // end of the rand part
    int rnd_ind = 0;

	for (int i = 0; i < NUM; i++) {
		// Generate a random number between 0 and 1
		randvec[i] = sfmt_to_res53(array[rnd_ind++]);  
	}
    free(array);
}

double roll_etta(double A, double theta, double W, double alpha){
    double a = sin((1-alpha)*theta) * pow((sin(alpha*theta)),(alpha/(1-alpha)));
    a = a / (pow(sin(theta), (1/(1-alpha))));
    double etta = pow((a / W),((1-alpha)/alpha));
    return etta * pow(A, 1.0 / alpha); // This is the case where A is not 1
}

// Function to draw one sample from Gaussian N(0, 2Dt) using Box-Muller
double sample_y(double u1, double u2, double t, double D) {
	//double D = 0.25; // for x and y direction in a 2d symmetric random walk
    // Standard normal via Box-Muller
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

    // Scale and shift
    return sqrt(2.0 * D * t) * z;
}

// Approximate jtheta2(0, q) with desired precision
double jtheta2_zero(double q, double epsilon) {
	// This is for the case where the cos argument is zero (z=0)
    if (q <= 0.0 || q >= 1.0) {
        return 0.0; // Invalid q
    }

    double sum = 0.0;
    int n = 0;

    while (1) {
        double term = pow(q, n * (n + 1));
        sum += term;

        if (term < epsilon) break;
        if (++n > 10000) break; // safeguard
    }

    return 2 * pow(q, 0.25) * sum;
}

void generate_random_matrix_and_relax(sfmt_t* sfmt, bool (*obs_or_not)[h_lattice * mesh_res], double p_occupation, int N_moves) {
    int total_elems = W_lattice * h_lattice + 2 * N_moves;

    int size = sfmt_get_min_array_size64(sfmt);
    if (size < total_elems) {
        size = total_elems;
        if (size % 2 != 0) {
            size++;  // Make sure it's even
        }
    }

    uint64_t* array;
	#if defined(__APPLE__) || (defined(__FreeBSD__) && __FreeBSD__ >= 3 && __FreeBSD__ <= 6)
		array = malloc(sizeof(uint64_t) * size);
		if (array == NULL) {
			printf("Can't allocate memory.\n");
			return;
		}
	#elif defined(_POSIX_C_SOURCE)
		if (posix_memalign((void **)&array, 16, sizeof(uint64_t) * size) != 0) {
			printf("Can't allocate memory.\n");
			return;
		}
	#else
		array = malloc(sizeof(uint64_t) * size);
		if (array == NULL) {
			printf("Can't allocate memory.\n");
			return;
		}
	#endif

    sfmt_fill_array64(sfmt, array, size);
	bool val; 
    int k = 0;
    for (int i = 0; i < W_lattice; i++) {
        for (int j = 0; j < h_lattice; j++) {
            val = sfmt_to_res53(array[k++]) < p_occupation;  // type: bool
				for (int dx = 0; dx < mesh_res; dx++){
					for (int dy = 0; dy < mesh_res; dy++){
						obs_or_not[mesh_res * i + dx][mesh_res * j + dy] = val;
					}
				}
        }
    }


	///////////////////////////////////////////////
	int capacity = W_lattice * h_lattice;
    int *x_position = malloc(capacity * sizeof(int));
    int *y_position = malloc(capacity * sizeof(int));

    int N_particles = 0;

    // Identify particle positions
    for (int i = 0; i < W_lattice; i++) {
        for (int j = 0; j < h_lattice; j++) {
            int x = i * mesh_res;
            int y = j * mesh_res;
            if (obs_or_not[x][y]) {
                x_position[N_particles] = x;
                y_position[N_particles] = y;
                N_particles++;
            }
        }
    }
	
	// Count mass and print array as 1s and 0s
    int mass_before = 0;
    for (int i = 0; i < W_lattice * mesh_res; i++) {
        for (int j = 0; j < h_lattice * mesh_res; j++) {
            if (obs_or_not[i][j]) {
                mass_before++;
                //printf("1 ");
            } else {
                //printf("0 ");
            }
        }
        //printf("\n"); // new line after each row
    }
	//printf("The mass before: %d\n", mass_before);

    int N_rejected_moves = 0;
    int particle_id, x0, y0, x1, y1, dx, dy;
    int x_check, y_check, x_new, y_new, x_old_col, x_new_col, y_old_row, y_new_row;
	bool blocked;
	struct step current_step;
	
	for (int move_id = 0; move_id < N_moves; move_id++) {

        particle_id = (int)(sfmt_to_res53(array[k++]) * N_particles);
        x0 = x_position[particle_id];
        y0 = y_position[particle_id];
        x1 = x0 + mesh_res - 1;
        y1 = y0 + mesh_res - 1;

        dx = 0;
        dy = 0;

        current_step = make_step(sfmt_to_res53(array[k++]), 0.0);
        dx = current_step.x;
        dy = current_step.y;

        blocked = false;

        if (dx > 0) { // move right
            if (x1 + dx >= W_lattice * mesh_res) {
                blocked = true;
                N_rejected_moves++;
            } else {
                x_new = x1 + dx;
                for (int j = 0; j < mesh_res; j++) {
                    y_check = y0 + j;
                    if (obs_or_not[x_new][y_check]) {
                        blocked = true;
                        N_rejected_moves++;
                        break;
                    }
                }
                if (!blocked) {
                    x_old_col = x0;
                    x_new_col = x1 + dx;
                    for (int j = 0; j < mesh_res; j++) {
                        obs_or_not[x_new_col][y0 + j] = true;
                        obs_or_not[x_old_col][y0 + j] = false;
                    }
                }
            }
        } 
        else if (dx < 0) { // move left
            if (x0 + dx < 0) {
                blocked = true;
                N_rejected_moves++;
            } else {
                x_new = x0 + dx;
                for (int j = 0; j < mesh_res; j++) {
                    y_check = y0 + j;
                    if (obs_or_not[x_new][y_check]) {
                        blocked = true;
                        N_rejected_moves++;
                        break;
                    }
                }
                if (!blocked) {
                    x_old_col = x1;
                    x_new_col = x0 + dx;
                    for (int j = 0; j < mesh_res; j++) {
                        obs_or_not[x_new_col][y0 + j] = true;
                        obs_or_not[x_old_col][y0 + j] = false;
                    }
                }
            }
        } 
        else if (dy > 0) { // move up
            if (y1 + dy >= h_lattice * mesh_res) {
                blocked = true;
                N_rejected_moves++;
            } else {
                int y_new = y1 + dy;
                for (int i = 0; i < mesh_res; i++) {
                    x_check = x0 + i;
                    if (obs_or_not[x_check][y_new]) {
                        blocked = true;
                        N_rejected_moves++;
                        break;
                    }
                }
                if (!blocked) {
                    y_old_row = y0;
                    y_new_row = y1 + dy;
                    for (int i = 0; i < mesh_res; i++) {
                        obs_or_not[x0 + i][y_new_row] = true;
                        obs_or_not[x0 + i][y_old_row] = false;
                    }
                }
            }
        } 
        else { // dy < 0 (move down)
            if (y0 + dy < 0) {
                blocked = true;
                N_rejected_moves++;
            } else {
                y_new = y0 + dy;
                for (int i = 0; i < mesh_res; i++) {
                    x_check = x0 + i;
                    if (obs_or_not[x_check][y_new]) {
                        blocked = true;
                        N_rejected_moves++;
                        break;
                    }
                }
                if (!blocked) {
                    y_old_row = y1;
                    y_new_row = y0 + dy;
                    for (int i = 0; i < mesh_res; i++) {
                        obs_or_not[x0 + i][y_new_row] = true;
                        obs_or_not[x0 + i][y_old_row] = false;
                    }
                }
            }
        }

        if (!blocked) {
            x_position[particle_id] += dx;
            y_position[particle_id] += dy;
        }
    }
    // Count mass and print array as 1s and 0s
    int mass_after = 0;
    for (int i = 0; i < W_lattice * mesh_res; i++) {
        for (int j = 0; j < h_lattice * mesh_res; j++) {
            if (obs_or_not[i][j]) {
                mass_after++;
                //printf("1 ");
            } else {
                //printf("0 ");
            }
        }
        //printf("\n"); // new line after each row
    }
	//printf("The mass after: %d\n", mass_after);
	printf("Checking conservation of mass, the difference is: %d\n", mass_after - mass_before);

    printf("Total rejected moves: %d\n", N_rejected_moves);

    free(x_position);
    free(y_position);
	///////////////////////////////////////////
    free(array);
}

void generate_random_matrix(sfmt_t* sfmt, bool (*obs_or_not)[h_lattice * mesh_res], double p_occupation) {
    int total_elems = W_lattice * h_lattice;

    int size = sfmt_get_min_array_size64(sfmt);
    if (size < total_elems) {
        size = total_elems;
        if (size % 2 != 0) {
            size++;  // Make sure it's even
        }
    }

    uint64_t* array;
	#if defined(__APPLE__) || (defined(__FreeBSD__) && __FreeBSD__ >= 3 && __FreeBSD__ <= 6)
		array = malloc(sizeof(uint64_t) * size);
		if (array == NULL) {
			printf("Can't allocate memory.\n");
			return;
		}
	#elif defined(_POSIX_C_SOURCE)
		if (posix_memalign((void **)&array, 16, sizeof(uint64_t) * size) != 0) {
			printf("Can't allocate memory.\n");
			return;
		}
	#else
		array = malloc(sizeof(uint64_t) * size);
		if (array == NULL) {
			printf("Can't allocate memory.\n");
			return;
		}
	#endif

    sfmt_fill_array64(sfmt, array, size);
	bool val; 
    int k = 0;
    for (int i = 0; i < W_lattice; i++) {
        for (int j = 0; j < h_lattice; j++) {
            val = sfmt_to_res53(array[k++]) < p_occupation;  // type: bool
				for (int dx = 0; dx < mesh_res; dx++){
					for (int dy = 0; dy < mesh_res; dy++){
						obs_or_not[mesh_res * i + dx][mesh_res * j + dy] = val;
					}
				}
        }
    }

    free(array);
}

void combineArrays(double* array1, int size1, double* array2, int size2, double* combined_array) {
    // Copy the contents of array1 to the combined_array
    memcpy(combined_array, array1, size1 * sizeof(double));

    // Copy the contents of array2 to the combined_array after array1
    memcpy(combined_array + size1, array2, size2 * sizeof(double));
}

void writeIntArrayToBinaryFile(int *array, int numElements, const char *fileName) {
    // Open the binary file in write mode ("wb")
    FILE *file = fopen(fileName, "wb");

    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    // Write the integer array to the binary file
    fwrite(array, sizeof(int), numElements, file);

    // Close the file
    fclose(file);
}

double *load_file_to_memory(const char *filename, long *n) {
    FILE *f = fopen(filename, "rb");
    fseek(f, 0, SEEK_END);
    long filesize = ftell(f);
    rewind(f);

    *n = filesize / sizeof(double);
    double *data = malloc(filesize);
    size_t readcount = fread(data, sizeof(double), *n, f);
    fclose(f);
    return data;
}

void writeDoubleArrayToBinaryFile(double *array, int numElements, const char *fileName) {
    // Open the binary file in write mode ("wb")
    FILE *file = fopen(fileName, "wb");

    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    // Write the double array to the binary file
    fwrite(array, sizeof(double), numElements, file);

    // Close the file
    fclose(file);
}

int generate_random_seed() {
    srand(time(NULL));  // Seed the random number generator with the current time
    return rand();
}

double FPT(double t, double x0, double L, double D) {
	// This function is the biscaling approach as described in the paper by Baravi, Barkai, Kessler, PRL, (2025)
	// We return the log of the result to avoid extreme values truncation errors
    double tau_L = L * L / D;
    double tau_0 = x0 * x0 / D;

    double log_delta = 1e-2;
    double delta = pow(10.0, log_delta);

    double log_t = log10(t);
    double log_t_vec[3] = {
        log_t - log_delta,
        log_t,
        log_t + log_delta
    };

    double t_vec[3] = {
        t / delta,
        t,
        t * delta
    };

    double jtheta2[3];
    double log_jtheta2[3];
    double epsilon = 1e-12; // precision for jtheta2 calculation

    for (int i = 0; i < 3; i++) {
        double q = exp(-M_PI * M_PI * t_vec[i] / tau_L);
        jtheta2[i] = jtheta2_zero(q, epsilon);
		
		// This --> jtheta2[i] might be zero (or so close to zero the machine can't tell the difference)
        if (jtheta2[i] == 0.0) {
            return 1.0;  // early exit
        }

        log_jtheta2[i] = log10(jtheta2[i]);
    }

    int i = 1;
    double dlogj_dlogt = (log_jtheta2[i+1] - log_jtheta2[i-1]) /
                         (log_t_vec[i+1] - log_t_vec[i-1]);
                         
	//if (dlogj_dlogt >= 0.0) {
        //// Log of negative number not allowed
        //return 1.0;
    //}

    double log_djtheta2_dtau = log_jtheta2[i] - log_t + log10(-dlogj_dlogt);
    double log_exp_term = -x0 * x0 / (4.0 * D * t) / log(10.0);  // log10(e^(-...)) = -... / ln(10)

    return log10(x0 / L) + log_djtheta2_dtau + log_exp_term;
}


double levy_pdf(double x, double A) {
	// This is the one sided levy pdf with alpha = 0.5
    if (x <= 0.0) {
        return 0.0;
    }
    // Constants
    double alpha = 0.5;
    // Scale parameter
    double scale = pow(cos(alpha * M_PI / 2.0), 1.0 / alpha) * pow(A, 1.0 / alpha);
    return sqrt(scale / (2.0 * M_PI)) * exp(-scale / (2.0 * x)) / pow(x, 1.5);
}

void run_sim(int iteration_thread, int seed, int particle_size, double F, double p_occupation, const char *save_folder){ //, double q
	//defining variables for the return time shortcut (biscaling approach)
	double D = 0.25; // for x and y direction in a 2d symmetric random walk
	double x0_reset = 20.0; //this is fixed to save as much time as possible
	double sampled_t, ft;
	double return_time;
	double L = CELL_HALF_WIDTH * mesh_res + 1 - particle_size;
	double A = fabs(tgamma(-0.5)) * x0_reset / sqrt(4.0 * M_PI * D);
	double c = 1.9; // fine tuned constant for the proposed pdf to be as close as possible to the wanted pdf
	
	// Other variables
	sfmt_t sfmt;
	sfmt_init_gen_rand(&sfmt, seed);
	double N_steps = pow(10.0, N_STEPS_LOG10); // length of each trajectory, converting from log10 to real number of steps
	int h = h_lattice * mesh_res;
	int W = W_lattice * mesh_res;
	int cell_half_width = CELL_HALF_WIDTH * mesh_res;
	
    //////////////////////////
    //memory defined on heap, should be faster.
	double *randvec = (double *)malloc((STEP_GENERATION_BATCH_SIZE)* sizeof(double)); // This is used to generate the trajectory.

	double *exit_times = (double *)malloc(NUM_traj * sizeof(double)); // The time is actually an integer number of steps but we have values > int32
	for (int i = 0; i < NUM_traj; i++) {
		exit_times[i] = -1;
	}
	bool (*obs_or_not)[h_lattice * mesh_res] = malloc(W_lattice * mesh_res * sizeof(bool[h_lattice * mesh_res])); //Initialize obstacle matrix
    if (randvec == NULL) {
        perror("Memory allocation failed");
    }
    ///////////////////////
    //The relaxation part
    
    // For the case of 5% relaxation
    //double percent_z = 5.0;
    //int N_moves = (int)round(226.19 * exp(4.379 * p_occupation));
    
    // For the case of 10% relaxation
	double percent_z = 10.0;
    int N_moves = (int)round(431.626 * exp(4.814 * p_occupation));
    //////////////////////////////
    
	generate_random_matrix_and_relax(&sfmt, obs_or_not, p_occupation, N_moves); //populate obstacle matrix
	
    for (int traj_id = 0; traj_id < NUM_traj; traj_id++) {
		generate_randvec(&sfmt, randvec); //populate steps with random values
		int rand_id  = 0;
		
		int temp_x, temp_y;
		struct step current_step;
		
		int x = (int)((-cell_half_width + particle_size) * randvec[rand_id++]);
		//int x = -1;
		int y = (int)(h * randvec[rand_id++]);
		if (y >= h){
			y = h - 1;
		}
		
		//for adding and removing lattice sites
		int lattice_x, lattice_y;
		
		double total_steps = 0;
		
		while (total_steps < N_steps) {
			total_steps = total_steps + 1.0;
			current_step = make_step(randvec[rand_id++], F);
			if (rand_id > STEP_GENERATION_BATCH_SIZE - 10){
				rand_id = 0;
				generate_randvec(&sfmt, randvec); //populate steps with random values
			}
			temp_x = x + current_step.x;
			temp_y = y + current_step.y;
			
			//shortcut part
			
			// The particle reached -x0_reset by taking a step to the left, we reset its position to x = 0
			if (x > -x0_reset && temp_x < -x0_reset + 1) { 
				while (1) {
					if (rand_id > STEP_GENERATION_BATCH_SIZE - 10){
						rand_id = 0;
						generate_randvec(&sfmt, randvec); //populate steps with random values
					}
					sampled_t = roll_etta(A, randvec[rand_id++] * M_PI, -log(randvec[rand_id++]), 0.5);
					ft = FPT(sampled_t, x0_reset, L, D);
					if (ft > 0.0) continue;
					
					if (log10(randvec[rand_id++] * c * levy_pdf(sampled_t, A)) < ft) {
						return_time = sampled_t;  // Accept the sample
						break; // step out of the "finding a return time" loop
					}
				}
				
				x = 0; // reset
				y = temp_y + (int) floor(sample_y(randvec[rand_id++], randvec[rand_id++], return_time, D));
				// check that y is not outside of the domain
				y = y % h;
				if (y < 0) y += h;
				total_steps = total_steps + return_time;
				
				if (total_steps > N_steps){
					total_steps = N_steps;
				}
				
				if (return_time < 0){
					printf("negetive return time");
				}
				continue;
			}
			
			//check for boundary conditions
			if (temp_x - particle_size + 1 < -(cell_half_width - 1)) { // Reflective, if the left most edge is out of the boundary we stay in place
				continue;  // this assumes you're inside a loop
			}
			if (temp_y > h - 1) {
				temp_y = temp_y - h;
			}
			if (temp_y < 0) {
				temp_y = temp_y + h;
			}
			
			if (temp_x > 0 && temp_x < W + 1) { // are we inside the wall
				// position of bottom right corner (temp_x - 1, temp_y)
				int x0 = temp_x - 1; // since the wall starts at x = 1 and array values index starts at 0
				int y0 = temp_y;
				// upper left corner
				int x1 = x0 - particle_size + 1;
				int y1 = temp_y + particle_size - 1;
				int wrap_y;  // always between 0 and h-1

				bool blocked = false;

				// is the position we jump to occupied (even partially) by an obstacle site
				// Check only the new edge the particle will occupy
				if (current_step.x > 0) { // moving right — check rightmost vertical edge
					for (int j = 0; j < particle_size; j++) {
						wrap_y = ((y0 + j) % h + h) % h;
						if (obs_or_not[x0][wrap_y] > 0) {
							blocked = true;
						}
					}
				} 
				else if (current_step.x < 0) { // moving left — check leftmost vertical edge
					for (int j = 0; j < particle_size; j++) {
						wrap_y = ((y0 + j) % h + h) % h;
						if (x1 > -1) {
							if (obs_or_not[x1][wrap_y] > 0) {
								blocked = true;
							}
						}
					}
				} 
				else if (current_step.y > 0) { // moving up — check topmost horizontal edge
					for (int j = 0; j < particle_size; j++) {
						wrap_y = (y1 % h + h) % h;
						if (x1 + j > -1) {
							if (obs_or_not[x1 + j][wrap_y] > 0) {
								blocked = true;
							}
						}
					}
				} 
				else if (current_step.y < 0) { // moving down — check bottommost horizontal edge
					for (int j = 0; j < particle_size; j++) {
						if (x1 + j > -1) {
							if (obs_or_not[x1 + j][y0] > 0) {
								blocked = true;
							}
						}
					}
				}
				if (blocked) {
					continue; // to next step
				}
			}
			//If we did not hit a boundary or an obstacle we make the step
			x = temp_x;
			y = temp_y;
			
			if (x > W) {
				// we escaped
				exit_times[traj_id] = total_steps;
				break;
			}
			if (W - x > N_steps - total_steps) { 
				// if the distance to the absorbing boundary is larger than the amount of steps left
				//printf("Didn't finish");
				break;
			}
		}
		//printf("%.3lf \n", exit_times[traj_id]);
		
	}
	//save to file
	char filename[300];
	int numElements = NUM_traj;
	
	// Get current time
	time_t now = time(NULL);
	struct tm t;
	localtime_r(&now, &t);   // <-- reentrant version

	// Format timestamp as YYYYMMDD_HHMMSS
	char timestamp[64];
	strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", &t);

	// Add timestamp to filename
	sprintf(filename, "%s/data_batch_%d_p_%.3lf_logN_%.2lf_%s.bin",
			save_folder,
			iteration_thread,
			p_occupation,
			N_STEPS_LOG10,
			timestamp);
			
	writeDoubleArrayToBinaryFile(exit_times, numElements, filename);
	
	free(randvec);
	free(obs_or_not);
	free(exit_times);
}

void make_save_folder(const char *full_path)
{

    struct stat st = {0};
    if (stat(full_path, &st) == -1) {
        // Folder does not exist — create it
        if (mkdir(full_path, 0777) != 0) {
            perror("Error creating directory");
            exit(EXIT_FAILURE);
        }
        printf("Directory created: %s\n", full_path);
    } else {
        printf("Directory already exists: %s\n", full_path);
    }
}

int main(int argc, char* argv[]) {
    // setting the constants
    double F = 0.0;
    double p_array[] = {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.51, 0.54, 0.57, 0.6, 0.65, 0.7, 0.73, 0.76, 0.79, 0.85, 0.9, 0.95};
    int p_array_length = 24;
	
	int particle_size_array[] = {4, 5};
	int particle_size_array_length = 2;
	
	int relax_p = 10;
    
    int Iterations = NUM_BATCHES;
    
    int starting_seed = generate_random_seed();  // Generate a random seed
	int seed_array[Iterations];
	
	sfmt_t sfmt;
	sfmt_init_gen_rand(&sfmt, starting_seed);
	
	for (int ind_p = 0; ind_p < p_array_length; ind_p++){
		double p_occupation = p_array[ind_p];
		
		for (int ind = 0; ind < particle_size_array_length; ind++){
			int particle_size = particle_size_array[ind];
			
			// Creating a folder in save_path to save the data to, if it's not already created. full_path is the save_path + the folder we create.
			char full_path[300];
			snprintf(full_path, sizeof(full_path),
             "%s/p_%.3lf_W_%d_relax_%d_periodic_y_particle_size_%d_mesh_%d",
            save_path, p_occupation, W_lattice, relax_p, particle_size, mesh_res);
			make_save_folder(full_path);
			
			for (int j = 0; j < Iterations; j++) {
				seed_array[j] = sfmt_genrand_uint32(&sfmt);
			}
			#pragma omp parallel for
			for (int i = 0; i < Iterations; i++) {
				run_sim(i, seed_array[i], particle_size, F, p_occupation, full_path);
			}
			#pragma omp barrier // Add a barrier to synchronize threads
		}
	}
    return 0;   
}













