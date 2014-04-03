/* Optimizaton 3
   Blocked matrix multiplication
   Modified by Pranami Bhattacharya
   Date: 02-17-14
   Referred: "The Cache Performance and Optimizations of Blocked Algorithms"
              M Lam et al, ASPLOS 4, 1991
*/

// matrix multiply program

#include <stdio.h>
#include <stdlib.h>

#define N	512
#define BLOCK_SIZE 2 // we divide the 512*512 matrix into blocks of 2*2
#define nn 256 // N/BLOCK_SIZE
// C = A x B

void matrix_multiply (double A[N][N], double B[N][N], double C[N][N]) {
	int i, j, k, jj, kk;
	int en = BLOCK_SIZE * (N/BLOCK_SIZE);
	for (i=0; i<N; i++)
		for (j=0; j<N; j++)
			C[i][j] = 0.0;

	for (kk=0; kk<en; kk+=BLOCK_SIZE) {
		for (jj=0; jj<en; jj+=BLOCK_SIZE) {
			for (i=0; i<N; i++) {
				//register double c__00_reg = 0;
				for (j=jj; j<(jj+BLOCK_SIZE); j++) {
					double temp = C[i][j];
					for (k=kk; k<(kk+BLOCK_SIZE); k++) {
						temp += A[i][k] * B[k][j];
					}
					C[i][j] = temp;
				}
			}
		}
	}
}

// read a matrix from standard input

void read_matrix (double A[N][N]) {
	int i, j;

	for (i=0; i<N; i++) {
		double d;
		int n;
		for (j=0; j<N; j++) {
			n = scanf ("%lf", &d);
			if (n != 1) {
				fprintf (stderr, "double expected!\n");
				exit (1);
			}
			A[i][j] = d;
		}
	}
}

// print a matrix to standard output

void print_matrix (double A[N][N]) {
	int i, j;

	for (i=0; i<N; i++) {
		for (j=0; j<N; j++) printf ("%f ", A[i][j]);
		printf ("\n");
	}
}

double X[N][N], Y[N][N], Z[N][N];

int main () {

	// read two N x N matrices

	read_matrix (X);
	read_matrix (Y);

	// multiply the matrices 

	matrix_multiply (X, Y, Z);

	// output the resulting matrix

	print_matrix (Z);
	return 0;
}

