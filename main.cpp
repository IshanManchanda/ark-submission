#pragma GCC target("avx,avx2,fma")

#include <bits/stdc++.h>
#include <iostream>
#include <omp.h>
using namespace std;

#define n 1000
//#define BLOCK_SIZE (n / 2)
//const int n = 1000;

long long costMatrixA[n][n];
long long costMatrixB[n][n];
long long productMat[n][n];

// Matrices to hold the required costs
long long minCostMatrixA[n][n];
long long maxCostMatrixB[n][n];

int main() {
	int i, j, k;
	srand(time(0));

	// Initialization
	// No benefit on separating out the initialization into different loops
	for (i = 0; i < n; ++i) {
		for (j = 0; j < n; ++j) {
			costMatrixA[i][j] = 1 + rand() % 10;
			costMatrixB[i][j] = 1 + rand() % 10;
			productMat[i][j] = 0;
		}
	}

	// We will use an iterative bottom-up DP approach to generate the required
	// min and max cost matrices instead of the top-down recursive approach.

	// Cost for last element
	minCostMatrixA[n - 1][n - 1] = costMatrixA[n - 1][n - 1];
	// Cost for remaining elements in the last row
	for (j = n - 2; j >= 0; --j)
		minCostMatrixA[n - 1][j] = costMatrixA[n - 1][j] + minCostMatrixA[n - 1][j + 1];

	for (i = n - 2; i >= 0; --i) {
		// Cost for last column in row i
		minCostMatrixA[i][n - 1] = costMatrixA[i][n - 1] + minCostMatrixA[i + 1][n - 1];

		// Cost for all other elements in row i
		for (j = n - 2; j >= 0; --j)
			minCostMatrixA[i][j] = costMatrixA[i][j] + \
			min(minCostMatrixA[i + 1][j], minCostMatrixA[i][j + 1]);
	}

	// Same procedure for B, we do them separately to maximize chance of cache hits
	// Costs for last element and last row
	maxCostMatrixB[n - 1][n - 1] = costMatrixB[n - 1][n - 1];
	for (j = n - 2; j >= 0; --j)
		maxCostMatrixB[n - 1][j] = costMatrixB[n - 1][j] + maxCostMatrixB[n - 1][j + 1];

	for (i = n - 2; i >= 0; --i) {
		// Cost for last column
		maxCostMatrixB[i][n - 1] = costMatrixB[i][n - 1] + maxCostMatrixB[i + 1][n - 1];

		for (j = n - 2; j >= 0; --j)
			maxCostMatrixB[i][j] = costMatrixB[i][j] + \
			max(maxCostMatrixB[i + 1][j], maxCostMatrixB[i][j + 1]);
	}

	// Block multiplication algorithm
	// Both with and without parallelization it works slower than the parallel
	// naive method (with the locality of reference optimization).
//#pragma omp parallel for
//	for (ii = 0; ii < n; ii += BLOCK_SIZE)
//		for (kk = 0; kk < n; kk += BLOCK_SIZE)
//			for (jj = 0; jj < n; jj += BLOCK_SIZE)
//				for (i = ii; i < ii + BLOCK_SIZE; i++)
//					for (k = kk; k < kk + BLOCK_SIZE; k++)
//						for (j = jj; j < jj + BLOCK_SIZE; j++)
//							productMat[i][j] += minCostMatrixA[i][k] * maxCostMatrixB[k][j];

	// Computing the product matrix in parallel with the naive algorithm.
	// Except the order of the last 2 loops has been changed to leverage
	// locality of reference for maximum cache hits.
#pragma omp parallel for
	for (i = 0; i < n; ++i)
		for (k = 0; k < n; ++k)
			for (j = 0; j < n; ++j)
				productMat[i][j] += minCostMatrixA[i][k] * maxCostMatrixB[k][j];

	// Initialize the filter of size 4 x n
	long long filterArray[4][n];
	for (i = 0; i < 4; ++i)
		for (j = 0; j < n; ++j)
			filterArray[i][j] = rand() % 2;

	// Matrix of dimension (n/c) x 1 where c = 4
	long long finalMat[n / 4];

	// Applying the filter in parallel
#pragma omp parallel for
	for (i = 0; i < n - 4; i += 4) {
		long long sum = 0;

		// Dot product of 4xn portion of productMat
		// Here we use the same locality of reference optimization
		for (int filterRow = 0; filterRow < 4; filterRow++)
			for (j = 0; j < n; ++j)
				sum += productMat[i + filterRow][j] * filterArray[filterRow][j];

		finalMat[i / 4] = sum;
	}

	return 0;
}
