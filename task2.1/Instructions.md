  ## Problem Statement

  `costMatrixA` and `costMatrixB` are two `n` x `n` matrices given to you. Each cell in these matrices contains the cost you have to pay on landing on that cell.

  You have two functions `FindMinCostA()` and `FindMaxCostB()`.

  * `FindMinCostA()` which returns the minimum cost of going from cell `i,j` to cell `n,n` in `costMatrixA`. This cost is the sum of the costs of the cells of `costMatrixA` which will be on your path from `i,j` to `n,n`. This path will be the minimum cost path out of all possible paths.

 * `FindMaxCostB()` which returns the maximum cost of going from cell `i,j` to cell `n,n` in `costMatrixB`. This cost is the sum of the costs of the cells of `costMatrixB` which will be on your path from `i,j` to `n,n`.This path will be the minimum cost path out of all possible paths.
  
  `n` will be always be divisible by `4`. 
  
  These functions will work for small values of `n` like `3` or `4`. However for higher values of `n` like `100` they will fail.
  Start by optimising these functions to work for `n <= 1000. 

  >HINT 1: For this part, you can use additional storage and think about ways to avoid repetition in recursive calls.

  For the next part, we have a product matrix prductMat. 
  prductMat[i][j] stores the value of
  
  `FindMinCostA(i,0)*FindMaxCostB(0,j,n) + FindMinCostA(i,1,n)*FindMaxCostB(1,j,n).... + FindMinCostA(i,n-1,n)*FindMaxCostB(n-1,j,n)`
  
  >HINT 2: We have used the common matrix multiplication code to do this. However, this can be further optimised to  maximize the chances of 
  sequential element access.

  After obtaining the product matrix, we apply a basic a filter on it. If the filter's dimension is `c` x `n`, then we replace
  the dot product of this filter corresponding `c` rows of `prductMat` with a single element in a new matrix whose dimension is `(n/c)` x `1`.

  For example:
  
  The product matrix is `[[1,2,3,4],[2,3,4,5]]` 
  
  The filter is `[[1,2,1,2],[2,1,2,1]]`

  Then, after applying the filter, the `1` x `1` matrix will be 
  
  `[1.1  +  2.2  + 3.1 +  4.2  + 2.2 + 3.1 + 4.2 + 5.1] = [36]`

  The dimension of the filter given to you will be `4` x `n`
  
  >HINT 3: In this part, the same operation (dot product) is going to be repeated several times. Also, the operation here can easily
  be vectorised. Try to think of parallel computation based optimisations to optimise this portion. 

  *You will be marked on the extent to which you succeeded at optimising the code.*

### For getting your program's running time (on Linux/macOS)
* Add `time` before the command you want to measure on terminal
### For getting your program's running time (on Windows)
* Use Powershell to run your commands
* Put the given `timeit.txt` file given in any directory (except your project directory)
* Rename it to  `timeit.bat`
* Add its file location to the system PATH variable
* Run this command after building:
`timeit <<exe file name>> -f input.txt`