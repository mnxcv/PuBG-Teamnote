# PuBG Teamnote
## Contents
___
### Tips 

### Data_structures
- (Lazy) Segment Tree 
- Union-find(Disjoint Set)

### DP Techniques
- Hibye1217's tip
- Bitmask 
- Knapsack
- Tree DP

### Geometry
- CCW + Convex Hull 
- Line-segment intersection
- Minimum Enclosing Circle with Heuristic Alg.
- Rotating calipers

### Graphs
- 2-sat Problem
- Dijkstra
- LCA
- MST
- Topological sort

### Mathematics
- Chinese Remainder Theorem 
- Euler's Phi function
- Fast Fourier Transformation
- Fibonacci with Matrices
- FlT(Fermat's little Theorem)
- Mobius inversion formula
    - Mobius function
- Well-known Combinatorics sequences
    - Catalan Numbers
    - Stirling Numbers
    - 더 아는거 있으면 추가좀

### String
- kmp
- Manacher
- trie

### Others
- Sprague-Grundy

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
### Union-find(Disjoint Set)
    int par[1000010];
    int depth[1000010];

    void init() {
        for (int i = 0; i < 1000010; i++) {
            par[i] = i;
        }
    }

    int ans(int x) {
        if (par[x] == x) return x;
        else return par[x] = ans(par[x]);
    }

    void uni(int x, int y) {
        if (ans(x) == ans(y)) return;
        else {
            int ax = ans(x);
            int ay = ans(y);
            if (depth[ax] < depth[ay]) {
                par[ax] = ay;
            }
            else {
                par[ay] = ax;
            }
            if (depth[ax] == depth[ay]) {
                depth[ax]++;
            }
        }
    }
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

### CCW + Convex Hull
    #include <bits/stdc++.h>
    using namespace std;

    #define X first
    #define Y second
    #define PRECISION 0

    using ll = long long;
    using ld = long double;

    using point = pair<ll,ll>;
    using vec = pair<ll,ll>;

    int sgn(ll x){return (x > 0) - (x < 0);}

    vec get_vector(point a, point b){
        return {b.X-a.X, b.Y-a.Y};
    }

    int ccw(vec u, vec v){
        ll cross_product = u.X*v.Y - u.Y*v.X;
        return sgn(cross_product);
    }

    int ccw(point p1, point p2, point p3){
        vec u = get_vector(p1, p2);
        vec v = get_vector(p2, p3);
        return ccw(u,v);
    }

    vec rev(vec v1){
        return {-1*v1.X, -1*v1.Y};
    }

    bool intersect(point p1, point p2, point p3, point p4){
        int ab = ccw(p1,p2,p3)*ccw(p1,p2,p4);
        int cd = ccw(p3,p4,p1)*ccw(p3,p4,p2);
        if(ab==0 && cd==0){
            return (min(p1,p2)<=max(p3,p4) && min(p3,p4)<=max(p1,p2));
        }
        return (ab <=0 && cd <=0);
    }

    ll ans;
    int n;
    point p[100005];

    int main(){
        ios::sync_with_stdio(0);
        cin.tie(0);
        cout.setf(ios::fixed); cout.precision(PRECISION);

        cin >> n;
        for(int i=0; i<n; i++){
            ll x, y;
            cin >> x >> y;
            p[i] = {x,y};
        }
        sort(p, p+n);
        vector<point>hull;
        for(int i=0; i<n; i++){
            if(hull.size()<2){
                hull.push_back(p[i]);
            }
            else{
                while(hull.size()>=2){
                    if(ccw(hull[hull.size()-1], hull[hull.size()-2], p[i])<=0){
                        hull.pop_back();
                    }
                    else{
                        break;
                    }
                }
                hull.push_back(p[i]);
            }
        }
        ans += hull.size();
        while(!hull.empty()) hull.pop_back();

        sort(p, p+n, greater<point>());
        for(int i=0; i<n; i++){
            if(hull.size()<2){
                hull.push_back(p[i]);
            }
            else{
                while(hull.size()>=2){
                    if(ccw(hull[hull.size()-1], hull[hull.size()-2], p[i])<=0){
                        hull.pop_back();
                    }
                    else{
                        break;
                    }
                }
                hull.push_back(p[i]);
            }
        }
        ans += hull.size();
        cout << ans - 2;
    }

### Line-segment intersection
    bool intersect(point p1, point p2, point p3, point p4){
        int ab = ccw(p1,p2,p3)*ccw(p1,p2,p4);
        int cd = ccw(p3,p4,p1)*ccw(p3,p4,p2);
        if(ab==0 && cd==0){
            return (min(p1,p2)<=max(p3,p4) && min(p3,p4)<=max(p1,p2));
        }
        return (ab <=0 && cd <=0);
    }
### Minimum Enclosing Circle with Heuristic Alg.
    pld coords[1005];

    ld dst(pld p1, pld p2){
        return (p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y);
    }

    int main(){
        ios::sync_with_stdio(0);
        cin.tie(0);
        cout.setf(ios::fixed); cout.precision(PRECISION);

        ld centerX = 0, centerY = 0, ratio=0.1, r;
        cin >> n;
        for(int i=0; i<n; i++){
            ld px, py;
            cin >> px >> py;
            centerX += px;
            centerY += py;
            coords[i]={px,py};    
        }
        //the mean position of all the coordinates on the plane
        centerX /= n;
        centerY /= n;
        //shuffling
        unsigned int seed = 0;
        shuffle(coords, coords+n, default_random_engine(seed));
        for(int i=0; i<=30000; i++){
            r = -1;
            int cursor = -1;
            for(int j=0; j<n; j++){
                ld newR = dst(coords[j],{centerX, centerY});
                if(r<newR){
                    r=newR;
                    cursor = j;
                }
            }
            centerX = centerX+(coords[cursor].x-centerX)*ratio;
            centerY = centerY+(coords[cursor].y-centerY)*ratio;
            ratio *= 0.999;
        }
        if((int)(centerX*1000)==0) centerX=0;
        if((int)(centerX*1000)==0) centerY=0;
        cout << centerX << ' ' << centerY << '\n' << sqrtl(r);
    }
### Rotating calipers

___
## Graphs

### 2-sat Problem

### LCA
    vector<int> tree[100010];
    int parent[100010][20];
    bool visited[100010];
    int depth[100010];

    signed main() {
        int n; cin >> n;
        for (int i = 1; i < n; i++) {
            int tmp1, tmp2;
            cin >> tmp1 >> tmp2;
            tree[tmp1].push_back(tmp2);
            tree[tmp2].push_back(tmp1);
        }
        //n
        queue<int> q;
        q.push(1);
        visited[1] = true;
        parent[1][0] = 1;
        depth[1] = 1;
        while (!q.empty()) {
            int tmp = q.front();
            q.pop();
            for (int& i : tree[tmp]) {
                if (!visited[i]) {
                    q.push(i);
                    visited[i] = true;
                    depth[i] = depth[tmp] + 1;
                    parent[i][0] = tmp;
                }
            }
        }
        //2*n
        for (int i = 1; i < 20; i++) {
            for (int j = 1; j <= n; j++) {
                parent[j][i] = parent[parent[j][i - 1]][i - 1];
            }
        }
        //n*20
        int m;
        cin >> m;
        for (int i = 0; i < m; i++) {
            int tmp1, tmp2;
            cin >> tmp1 >> tmp2;
            if (depth[tmp1] > depth[tmp2]) {
                int ascend = depth[tmp1] - depth[tmp2];
                int ascendtmp = ascend;
                for (int i = 0; i <= log2(ascend); i++) {
                    if (ascendtmp % 2) tmp1 = parent[tmp1][i];
                    ascendtmp /= 2;
                }
            }
            else if (depth[tmp1] < depth[tmp2]) {
                int ascend = depth[tmp2] - depth[tmp1];
                int ascendtmp = ascend;
                for (int i = 0; i <= log2(ascend); i++) {
                    if (ascendtmp % 2) tmp2 = parent[tmp2][i];
                    ascendtmp /= 2;
                }
            }
            while (tmp1 != tmp2) {
                debug(tmp1);
                debug(tmp2);
                for (int i = 0; i <= log2(depth[tmp1]) + 2; i++) {
                    if (parent[tmp1][i] == parent[tmp2][i]) {
                        if (!i) {
                            tmp1 = parent[tmp1][i];
                            tmp2 = parent[tmp2][i];
                            break;
                        }
                        else {
                            tmp1 = parent[tmp1][i - 1];
                            tmp2 = parent[tmp2][i - 1];
                            break;
                        }
                    }
                }
            }
            cout << tmp1 << '\n';

        }
    }
### MST
    int main(){
        ios::sync_with_stdio(0);
        cin.tie(0);
        cin >> v >> e;
        while(e--){
            int a, b, c;
            cin >> a >> b >> c;
            edges.push_back({c,{a,b}});
        }
        for(int i=1; i<=v; i++){
            parent[i] = i;
        }
        sort(edges.begin(),edges.end());
        for(auto cur:edges){
            int root1 = fnd(cur.st);
            int root2 = fnd(cur.en);
            if(root1==root2) continue;
            mst += cur.cost;
            uni(cur.st, cur.en);
        }
        cout << mst;
    }
### Topological Sort
    int n,m;

    vector<int>edges[32005];
    vector<int>ans;

    int inDegree[32005];

    int main(){
        ios::sync_with_stdio(0);
        cin.tie(0);
        cin >> n >> m;
        while(m--){
            int st, en;
            cin >> st >> en;
            edges[st].push_back(en);
            inDegree[en]++;
        }
        queue<int>q;
        for(int i=1; i<=n; i++){
            if(inDegree[i]==0) q.push(i);
        }
        while(!q.empty()){
            int cur = q.front();
            q.pop();
            ans.push_back(cur);
            for(auto nxt:edges[cur]){
                inDegree[nxt]--;
                if(inDegree[nxt]==0) q.push(nxt);
            }
        }
        for(int &i:ans) cout << i << ' ';
    }

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
    typedef struct pair<pair<long long int, long long int>, pair<long long int, long long int>> mat;

    mat I = make_pair(make_pair(0, 1), make_pair(1, 1));

    mat matip(mat matrix1, mat matrix2) {
        mat res;
        res.first.first = (matrix1.first.first * matrix2.first.first + matrix1.first.second * matrix2.second.first) % INT;
        res.first.second = (matrix1.first.first * matrix2.first.second + matrix1.first.second * matrix2.second.second) % INT;
        res.second.first = (matrix1.second.first * matrix2.first.first + matrix1.second.second * matrix2.second.first) % INT;
        res.second.second = (matrix1.second.first * matrix2.first.second + matrix1.second.second * matrix2.second.second) % INT;
        return res;
    }


    mat multiple(mat matrix, long long int k) {
        if (k == 1) return matrix;
        if (k % 2) return matip(I, multiple(matrix, k - 1));
        else {
            mat half = multiple(matrix, k / 2);
            return matip(half, half);
        }
    }

    int main() {
        ios_base::sync_with_stdio(false);
        cin.tie(NULL); cout.tie(NULL);

        long long int n;
        cin >> n;
        cout << multiple(I, n).second.first << '\n';
    }
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
