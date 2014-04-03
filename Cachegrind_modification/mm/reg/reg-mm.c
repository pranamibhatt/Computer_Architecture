/* Original matrix multiply program changed to assign the result to a register
   The register content again assigned to memory
   loop nest optimization
   Modified by Pranami Bhattacharya
   Date: 02-17-14
*/

// matrix multiply program

#include <stdio.h>
#include <stdlib.h>

#define N	512

// C = A x B

void matrix_multiply (double A[N][N], double B[N][N], double C[N][N]) {
	int i, j, k;
	register double c__00_reg = 0;  	
	for (i=0; i<N; i++) {
		for (j=0; j<N; j++) {
		//	C[i][j] = 0.0;
			c__00_reg = 0;
			for (k=0; k<N; k++) {
				c__00_reg += A[i][k] * B[k][j];
			}
		C[i][j] = c__00_reg;
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

