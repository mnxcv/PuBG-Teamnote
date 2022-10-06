# PuBG Teamnote
## Contents
___
### Tips 

### Data_structures
- Segment Tree
    - Lazy Segment Tree 
- Sparse Table
- Union-find(Disjoint Set)

### DP Techniques
- Hibye1217's tip
- Bitmask 
- Knapsack
- Tree DP <= 이거 두개는 난 모름;;

### Geometry
- CCW <= 너무 쉬운가?
- Convex Hull 
- Line-segment intersection
- Minimum Enclosing Circle with Heuristic Alg. <= 나오긴 할까?
- Rotating calipers <= 넣으면 쓸수 있나?
- 

### Graphs
- 2-sat Problem <= 난 몰루;;
- Dijkstra
- LCA
- MST
- 

### Mathematics
- Chinese Remainder Theorem 
- Euler's Phi function
- Fast Fourier Transformation
- Fibonacci with Matrices
- FlT(Fermat's little Theorem)
- Mobius inversion formula <= 이건 쓸수 있나?
    - Mobius function
- Well-known Combinatorics sequences
    - Catalan Numbers
    - Stirling Numbers
    - 더 아는거 있으면 추가좀

### String
- kmp <= ㅋㅋ;;
- Manacher <= ?
- trie <= 아는사람 있나?

### Others
- Sprague-Grundy <= ;;

___
## Tips
- 긴장하지 말것
- 모르겠으면 일단 Naive하게 접근해보기
- 대충 이런거 쓰는곳

___

## Data_Structures

### Segment Tree
    class segment_tree {
    private:
        vector<ll> tree;
        vector<ll> lazy;

    public:
        segment_tree(int N) {
            tree.resize(4*N, 0LL);
            lazy.resize(4*N, 0LL);
        }

        void init(int node, int start, int end){
            if(start==end){
                tree[node] = arr[start];
            }
            else{
                int mid = (start+end)>>1;
                init(2*node, start, mid);
                init(2*node+1, mid+1, end);
                tree[node] = tree[2*node] + tree[2*node+1];
            }
        }

        void lazy_prop(int start, int end, int node){
            if(lazy[node]){
                tree[node] += (end-start+1)*lazy[node];

                if(start!=end){
                    lazy[2*node] += lazy[node];
                    lazy[2*node+1] += lazy[node];
                }

                lazy[node]=0;
            }
        }

        void update(int start, int end, int node, int left, int right, ll val) {
            lazy_prop(start, end, node);

            if(left>end || right<start) return;

            if(start>=left && end<=right) {
                tree[node] += (end-start+1)*val;
                if(start!=end){
                    lazy[node*2] += val;
                    lazy[node*2+1] += val;
                }
                return;
            }
            int mid = (start+end)>>1;
            update(start, mid, 2*node, left, right, val);
            update(mid+1, end, 2*node+1, left, right, val);
            tree[node] = tree[node * 2] + tree[node * 2 + 1];
        }

        ll query(int start, int end, int node, int left, int right) {
            lazy_prop(start, end, node);
            if(left>end || right<start){
                return 0;
            }
            if(left<=start && right>=end){
                return tree[node];
            }
            int mid = (start+end)>>1;
            ll leftq = query(start, mid, 2*node, left, right);
            ll rightq = query(mid+1, end, 2*node+1, left, right);
            return leftq+rightq;
        }
    };
### Sparse Table

### Union-find(Disjoint Set)

___

## DP Techniques

### Hibye1217's tip

### Bitmask

### Knapsack

### Tree DP

___
## Geometry

### CCW

### Convex Hull

### Line-segment intersection

### Minimum Enclosing Circle with Heuristic Alg.

### Rotating calipers

___
## Graphs

### 2-sat Problem

### Dijkstra

### LCA

### MST
___
## Mathematics

### Chinese Remainder Theorem 

### Euler's Phi function

### Fast Fourier Transformation

참고 : long double 쓰다가 터질(TLE)수 있음

    typedef complex<ld> cpld;
    const ld PI = acosl(-1);
    void fft(vector<cpld>& series1, /*cpld w, */bool inverse = false, bool rounding = false) {
        int sz = series1.size();
        int revbit = 0;
        for (int i = 1; i < sz; i++) {
            int bit = sz / 2;
            while (((revbit ^= bit) & bit) == 0) bit /= 2;
            if (i < revbit) swap(series1[i], series1[revbit]);
        }
        for (int i = 1; i < sz; i *= 2) {
            ld x = PI * (-2 *inverse + 1) / i;
            cpld w = cpld(cos(x), sin(x));
            for (int j = 0; j < sz; j += i * 2) {
                cpld wp = cpld(1, 0);
                for (int k = 0; k < i; k++) {
                    cpld tmp = series1[i + j + k] * wp;
                    series1[i + j + k] = series1[j + k] - tmp;
                    series1[j + k] += tmp;
                    wp *= w;
                }
            }
        }
        if (inverse) {
            for (int i = 0; i < sz; i++) {
                series1[i] /= cpld(sz, 0);
                if(rounding) series1[i] = cpld(round(series1[i].real()), round(series1[i].imag()));
            }
        }
    }
    vector<cpld> multiple(vector<cpld>& v1, vector<cpld>& v2) {
        int n = 1;
        while (n <= v1.size() || n <= v2.size()) n <<= 1;
        n <<= 1;
        v1.resize(n); v2.resize(n);
        vector<cpld> res(n);
        cpld w = cpld(cos(2 * PI / n), sin(2 * PI / n));
        fft(v1); fft(v2);
        for (int i = 0; i < n; i++) {
            res[i] = v1[i] * v2[i];
        }
        fft(res, true, true);
        return res;
    }

### Fibonacci with Matrices

### FlT(Fermat's little Theorem)

### Mobius inversion formula

#### Mobius function

### Well-known Combinatorics sequences

#### Catalan Numbers

#### Stirling Numbers
___
## String

### kmp

### Manacher

### trie
___
## Others

### Sprague-Grundy
