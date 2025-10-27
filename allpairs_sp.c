// fox.c
// MPI implementation of min-plus matrix multiplication using Fox's algorithm
// plus repeated squaring to compute All-Pairs Shortest Paths.
// Compile: mpicc fox.c -o fox
// Run: mpirun -np P ./fox < input.txt

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

typedef long long ll;
const ll INF = (ll)1e12;

static inline ll min_ll(ll a, ll b) { return a < b ? a : b; }

// print matrix on root
void gather_and_print(ll *local, int N, int Q, int nloc, int rank, MPI_Comm cart_comm) {
    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    int world_rank;
    MPI_Comm_rank(cart_comm, &world_rank);

    int size;
    MPI_Comm_size(cart_comm, &size);

    // root rank assumed 0
    int root = 0;
    if (rank == root) {
        // allocate full matrix
        ll *full = (ll*)malloc(sizeof(ll) * N * N);
        // place own block
        int brow = coords[0], bcol = coords[1];
        for (int i=0;i<nloc;i++){
            for (int j=0;j<nloc;j++){
                full[(brow*nloc + i)*N + (bcol*nloc + j)] = local[i*nloc + j];
            }
        }
        // receive from others
        for (int r = 0; r < size; ++r) if (r != root) {
            int ccoords[2];
            MPI_Cart_coords(cart_comm, r, 2, ccoords);
            int rr = ccoords[0], rc = ccoords[1];
            ll *buf = (ll*)malloc(sizeof(ll) * nloc * nloc);
            MPI_Recv(buf, nloc*nloc, MPI_LONG_LONG_INT, r, 0, cart_comm, MPI_STATUS_IGNORE);
            for (int i=0;i<nloc;i++){
                for (int j=0;j<nloc;j++){
                    full[(rr*nloc + i)*N + (rc*nloc + j)] = buf[i*nloc + j];
                }
            }
            free(buf);
        }
        // print converting INF -> 0
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                ll v = full[i*N + j];
                if (v >= INF/2) v = 0;
                if (j) printf(" ");
                printf("%lld", v);
            }
            printf("\n");
        }
        free(full);
    } else {
        MPI_Send(local, nloc*nloc, MPI_LONG_LONG_INT, root, 0, cart_comm);
    }
}

// helper: allocate and zero
ll* alloc_mat(int n) {
    ll *m = (ll*)malloc(sizeof(ll) * n * n);
    if (!m) { fprintf(stderr, "malloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    for (int i=0;i<n*n;i++) m[i] = INF;
    return m;
}

// local min-plus multiply: C = min(C, A * B) where A,B are (nloc x nloc)
void local_minplus_mul(ll *A, ll *B, ll *C, int nloc) {
    // simple triple loop
    for (int i = 0; i < nloc; ++i) {
        for (int k = 0; k < nloc; ++k) {
            ll aik = A[i*nloc + k];
            if (aik >= INF/2) continue;
            for (int j = 0; j < nloc; ++j) {
                ll bkj = B[k*nloc + j];
                if (bkj >= INF/2) continue;
                ll sum = aik + bkj;
                if (sum < C[i*nloc + j]) C[i*nloc + j] = sum;
            }
        }
    }
}

// Distributed min-plus matrix multiplication using Fox algorithm.
// localA, localB: local blocks of input NxN matrices. localC: output local block.
// N: global matrix size. Q: process grid dimension. nloc = N/Q.
void fox_minplus(ll *localA, ll *localB, ll *localC, int N, int Q, int nloc, MPI_Comm cart_comm) {
    int rank;
    MPI_Comm_rank(cart_comm, &rank);
    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    int prow = coords[0], pcol = coords[1];

    // create row and column communicators
    MPI_Comm row_comm, col_comm;
    int remain_dims_row[2] = {0, 1}; // keep column varying? Actually for row communicator, keep row fixed => dims: [0,1]? We'll use MPI_Cart_sub
    int remain_dims_col[2] = {1, 0};
    MPI_Cart_sub(cart_comm, remain_dims_row, &row_comm);
    MPI_Cart_sub(cart_comm, remain_dims_col, &col_comm);

    ll *Atemp = (ll*)malloc(sizeof(ll) * nloc * nloc);
    // Initialize localC to INF
    for (int i=0;i<nloc*nloc;i++) localC[i] = INF;

    // working buffer for B shift
    MPI_Status status;

    for (int stage = 0; stage < Q; ++stage) {
        int root_col = (prow + stage) % Q; // column index of block to broadcast in this row
        if (pcol == root_col) {
            // copy localA to Atemp
            memcpy(Atemp, localA, sizeof(ll) * nloc * nloc);
        }
        // broadcast Atemp within the row
        MPI_Bcast(Atemp, nloc*nloc, MPI_LONG_LONG_INT, root_col, row_comm);

        // local multiply: C = min(C, Atemp * localB)
        local_minplus_mul(Atemp, localB, localC, nloc);

        // circular shift localB upwards by 1 in column (send to up, receive from down)
        int src_rank, dst_rank;
        // compute ranks in col_comm: row index changes
        int mycoords_col[2] = {prow, pcol};
        // In col_comm, ranks are ordered by row. Use MPI_Cart_shift on cart_comm with direction 0 (rows) and displacement 1
        MPI_Cart_shift(cart_comm, 0, 1, &src_rank, &dst_rank);
        // But src_rank/dst_rank are absolute ranks in cart_comm; we need to perform sendrecv_replace on those ranks
        MPI_Sendrecv_replace(localB, nloc*nloc, MPI_LONG_LONG_INT, dst_rank, 0, src_rank, 0, cart_comm, &status);
    }

    free(Atemp);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
}

// Scatter full matrix from root to blocks (each block to its process). root reads input and prepares blocks.
// On root: full matrix present in 'full' (row-major). On all others pass full==NULL.
void scatter_blocks(ll *full, ll *local, int N, int Q, int nloc, MPI_Comm cart_comm) {
    int rank;
    MPI_Comm_rank(cart_comm, &rank);
    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    if (rank == 0) {
        // send blocks to all processes (including root: copy)
        for (int r = 0; r < Q*Q; ++r) {
            int crd[2];
            MPI_Cart_coords(cart_comm, r, 2, crd);
            int brow = crd[0], bcol = crd[1];
            if (r == 0) {
                // copy root block
                for (int i=0;i<nloc;i++){
                    for (int j=0;j<nloc;j++){
                        local[i*nloc + j] = full[(brow*nloc + i)*N + (bcol*nloc + j)];
                    }
                }
            } else {
                // prepare buffer and send
                ll *buf = (ll*)malloc(sizeof(ll) * nloc * nloc);
                for (int i=0;i<nloc;i++){
                    for (int j=0;j<nloc;j++){
                        buf[i*nloc + j] = full[(brow*nloc + i)*N + (bcol*nloc + j)];
                    }
                }
                MPI_Send(buf, nloc*nloc, MPI_LONG_LONG_INT, r, 0, cart_comm);
                free(buf);
            }
        }
    } else {
        MPI_Recv(local, nloc*nloc, MPI_LONG_LONG_INT, 0, 0, cart_comm, MPI_STATUS_IGNORE);
    }
}

// Gather blocks from processes into full on root. (Opposite of scatter)
void gather_blocks(ll *local, ll *full, int N, int Q, int nloc, MPI_Comm cart_comm) {
    int rank;
    MPI_Comm_rank(cart_comm, &rank);
    if (rank == 0) {
        // copy own
        int coords[2]; MPI_Cart_coords(cart_comm, 0, 2, coords);
        int brow = coords[0], bcol = coords[1];
        for (int i=0;i<nloc;i++){
            for (int j=0;j<nloc;j++){
                full[(brow*nloc + i)*N + (bcol*nloc + j)] = local[i*nloc + j];
            }
        }
        // receive others
        int size; MPI_Comm_size(cart_comm, &size);
        for (int r = 1; r < size; ++r) {
            int ccoords[2]; MPI_Cart_coords(cart_comm, r, 2, ccoords);
            int rr = ccoords[0], rc = ccoords[1];
            ll *buf = (ll*)malloc(sizeof(ll) * nloc * nloc);
            MPI_Recv(buf, nloc*nloc, MPI_LONG_LONG_INT, r, 1, cart_comm, MPI_STATUS_IGNORE);
            for (int i=0;i<nloc;i++){
                for (int j=0;j<nloc;j++){
                    full[(rr*nloc + i)*N + (rc*nloc + j)] = buf[i*nloc + j];
                }
            }
            free(buf);
        }
    } else {
        MPI_Send(local, nloc*nloc, MPI_LONG_LONG_INT, 0, 1, cart_comm);
    }
}

// main
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Root reads the input
    int N = 0;
    ll *full_mat = NULL;
    if (world_rank == 0) {
        if (scanf("%d", &N) != 1) {
            fprintf(stderr, "Failed to read N\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        full_mat = (ll*)malloc(sizeof(ll) * N * N);
        for (int i=0;i<N;i++){
            for (int j=0;j<N;j++){
                long long v;
                if (scanf("%lld", &v) != 1) { fprintf(stderr, "bad input\n"); MPI_Abort(MPI_COMM_WORLD,1); }
                if (v == 0 && i != j) full_mat[i*N + j] = INF; // no edge
                else full_mat[i*N + j] = v;
            }
        }
    }

    // broadcast N to all
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // determine Q such that P = Q*Q
    int P = world_size;
    int Q = 0;
    // compute integer square root
    int qcheck = (int)(sqrt((double)P) + 0.5);
    if (qcheck * qcheck == P) Q = qcheck;
    else Q = 0;

    if (Q == 0) {
        if (world_rank==0) fprintf(stderr, "Error: number of processes P=%d is not a perfect square (P=Q*Q required).\n", P);
        MPI_Finalize();
        return 0;
    }
    if (N % Q != 0) {
        if (world_rank==0) fprintf(stderr, "Error: matrix dimension N=%d is not divisible by Q=%d (N mod Q == 0 required).\n", N, Q);
        MPI_Finalize();
        return 0;
    }

    int dims[2] = {Q, Q};
    int periods[2] = {1, 1}; // periodic for circular shifts
    int reorder = 1;
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);

    int rank;
    MPI_Comm_rank(cart_comm, &rank);
    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    int nloc = N / Q;

    // allocate local blocks
    ll *localA = (ll*)malloc(sizeof(ll) * nloc * nloc);
    ll *localB = (ll*)malloc(sizeof(ll) * nloc * nloc);
    ll *localC = (ll*)malloc(sizeof(ll) * nloc * nloc);
    if (!localA || !localB || !localC) { fprintf(stderr,"alloc fail\n"); MPI_Abort(MPI_COMM_WORLD,1); }

    // scatter initial full_mat into localA (initial matrix D1)
    scatter_blocks(full_mat, localA, N, Q, nloc, cart_comm);
    // For squaring, we'll need localB too; initially B = A
    memcpy(localB, localA, sizeof(ll) * nloc * nloc);

    // Free full_mat on root for now (we'll re-create when needed)
    if (world_rank == 0) {
        free(full_mat);
        full_mat = NULL;
    }

    // Repeated squaring: compute D = D ⊗ D, repeatedly until power >= N
    int power = 1;
    // We'll store current distributed matrix in localA
    MPI_Barrier(cart_comm);
    double t0 = MPI_Wtime();

    while (power < N) {
        // compute localC = localA ⊗ localA (using Fox)
        // set localB = localA for multiplication
        memcpy(localB, localA, sizeof(ll) * nloc * nloc);
        // zero localC
        for (int i=0;i<nloc*nloc;i++) localC[i] = INF;

        // Call Fox
        fox_minplus(localA, localB, localC, N, Q, nloc, cart_comm);

        // After multiplication, localC contains the squared block
        // replace localA <- localC
        memcpy(localA, localC, sizeof(ll) * nloc * nloc);

        power *= 2;
    }

    double t1 = MPI_Wtime();
    double elapsed = t1 - t0;

    // Optionally: gather elapsed time to root and print (as milliseconds)
    double local_time = elapsed * 1000.0; // ms
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    if (rank == 0) {
        fprintf(stderr, "Computation time (max across processes, ms, excluding I/O): %.3f\n", max_time);
    }

    // gather and print final matrix
    gather_and_print(localA, N, Q, nloc, rank, cart_comm);

    // cleanup
    free(localA); free(localB); free(localC);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
