#include <assert.h>
#include <stdio.h>

// Do the add vector operation
int* add(int* a, int* b, int* result, int N) {
    // -:YOUR CODE HERE:-
    for(size_t i=0; i<N; i++)
        result[i] = a[i] + b[i];
    return result;
}

// Configure the dynamic memory to initialize
// the variables a, b and result as pointers.

void onHost() {
    int N = 10;

    // -:YOUR CODE HERE:-
    int* a = (int*) malloc(N*sizeof(int));
    int* b = (int*) malloc(N*sizeof(int));
    int* result = (int*) malloc(N*sizeof(int));

    for (int i = 0; i < N; i++) {
        a[i] = -i;
        b[i] = i * i;
        result[i] = 0;
    }

    add(a, b, result, N);

    for (int i = 0; i < N; i++) {
        assert(a[i] + b[i] == result[i]);
    }

    printf("-: successful execution :-\n");

    // -:YOUR CODE HERE:-
}

int main() {
    onHost();
    return 0;
}