# PuBG Teamnote
## Contents
___
### Tips 

### Data_structures
- (Lazy) Segment Tree 
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

### (Lazy) Segment Tree
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
1. 본인이 본 적 있는 DP 문제인가?
yes: 그 때 어떻게 풀었는지 기억을 되살려보기. 전혀 기억 안 나면 1→no 로
no: 2번으로

2. 상태공간 정의해보기
수열에서: ...의 i번째 값 / [1, i] 구간에서... / [st, ed]에서...
트리에서: 정점 v를 루트로 하는 서브트리에서...
+ 추가: 필요한 추가 정보 (예로, RGB 거리의 경우 "색깔")

3. 상태공간을 토대로 답을 구해보기
보통은 위 4가지 경우에 대해 각각 DP[N] / DP[N] / DP[1][N] / DP[root]
또는 min(DP[i])나 max(DP[i]), sum(DP[i])일 수도?

4. 점화식 만들기
부분 문제들을 어떻게 합쳐야 할지 생각해보는 과정으로, 사실 여기가 가장 어려움.
DP[i]를 부분문제들 DP[x]의 조합으로 구해볼 수도 있고
DP[i]를 부분문제로 가지는 DP[x]들을 업데이트해볼 수도 있음.
아래는 실패 시 타봐야 하는 테크트리
점화식이 안 만들어진다면: 2번으로 돌아가서 상태공간의 매개변수를 조정해보기
근데 그래도 너무 안 된다면: 이 문제가 DP 문제인지 다시 한 번 생각해보기

+ 시간복잡도 계산해보기
DP의 시간복잡도는 초항 수 × 초항 계산에 걸리는 시간 + 상태공간의 수 × 점화식 계산에 걸리는 시간 이지만
초항은 아래 쓰여있듯이 보통 자명한 경우 ( O(1) )가 많아서 사실상
상태공간과 점화식만으로 시간복잡도를 계산해볼 수 있음.
시간이 오래 걸린다면: 혹시 top-down으로 탐색하는 상태공간의 수를 줄일 수 있는지 확인해보기
그렇지 않다면: 4번으로 돌아가서 점화식을 더 빠르게 수정해보기
도저히 안 줄어든다면: 2번으로 돌아가서 매개변수를 조정해보기
그래도 안 줄어든다면: 이 문제가 DP 문제인지 다시 한 번 생각해보기

5. 초항 구하기
이건 보통 어렵지 않음. 그냥 수학적 귀납법이 "N = 1일 때는 자명하다." 하고 넘기는 그런 부분.
아까 써놨던 4가지 경우에 대해서, 보통은 각각 DP[0] / DP[1] / DP[i][i] / DP[leaf]가 됨.
값은 웬만해서는 0 / 1 / A[i] 3개 중 하나.
가끔씩은 초항이 아니라 '초기 상태'를 놓는 게 더 편할 때도 있다.

6. 마지막 확인
시간복잡도를 다시 한 번 확인해보고, 경우에 따라 아래 생각을 추가적으로 해보기

그냥 N이 10^18 같은 정신 나간 수인 경우:
 - 선형점화식이라면: 행렬의 거듭제곱으로 DP를 계산해볼 수 있을까?
 - 아니라면: 혹시 선형점화식을 위해 2. 매개 변수를 조정하거나 / 4. 점화식을 재계산해봐야 할까?
 - 기타: 이거 사실 수학 문제가 아닐까?

(상태공간 × 점화식 계산)은 크지만, 대부분의 상태공간은 탐색할 필요가 없는 경우
 - 혹시 탑다운으로 AC를 받을 수 있지 않을까?

(난이도가 있는 경우) 문제의 난이도가 비정상적으로 어려운 경우
 - 그들만의 well-known DP optimization이 있는 게 아닐까?

시간 걱정은 딱히 없지만 메모리가 터질 거 같은 경우
(의외로 가끔씩 ll dp[5020][5020]; 같은 거에 이상한 자료구조 몇몇개 들어간다면 주의해봐야 함)
 - DP[i][\*]의 계산에 DP[i-1][\*]만 사용되는 등, 대부분의 값들을 계산에 사용하지 않는다면: 이를 Sliding Window로 처리하면 메모리가 줄어들 거 같은데?
 - 그런 거 없다면: 2. 매개변수 조정
 - 기타: 이거 사실 수학 문제가 아닐까 22 / 전처리가 정해가 아닐까 
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
    vector<int>pos;
    vector<int> computeTable(string s){
        int en = s.size();
        int i=0;
        vector<int> lps(en, 0);
        for(int j=1; j<en; j++){
            while(i>0 && s[i] != s[j]){
                i = lps[i-1];
            }
            if(s[i]==s[j]){
                lps[j] = ++i;
            }
        }
        return lps;
    }
    void kmp(string original, string tofind){
        vector<int>table = computeTable(tofind);
        int oSize = original.size();
        int tSize = tofind.size();
        int j=0;
        for(int i=0; i<oSize; i++){
            while(j > 0 && original[i]!=tofind[j]){
                j = table[j-1];
            }
            if(original[i] == tofind[j]){
                if(j == tSize-1){
                    pos.push_back(i-tSize+2);
                    j = table[j];
                }
                else{
                    j++;
                }
            }
        }
    }
### Manacher

### trie
___
## Others

### Sprague-Grundy
