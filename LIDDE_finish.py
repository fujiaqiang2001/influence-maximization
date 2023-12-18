import random
import networkx as nx
import matplotlib.pyplot as plt

"""
F:缩放因子
cr:交叉概率
g_max:迭代次数数
pop:种群大小
k:种子节点数量
up_a=η:上界的系数
up_b=θ:上界的偏移值
a=α:节点集合的原始影响力重要系数
b=β:节点集单跳影响力的重要系数
c=γ:节点集的两跳影响力重要系数
"""

F = 0.6
cr = 0.4
g_max = 200
pop = 10
k = 5
up_a = 10
up_b = 50
a = b = c = 1


def example():
    E = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 4), (3, 5), (5, 6), (6, 2), (6, 3), (6, 4)]
    G = nx.DiGraph()
    G.add_edges_from(E)
    pos = nx.shell_layout(G)
    nx.draw(G, pos, node_size=300, with_labels=True, node_color='blue')
    # plt.show()
    return G


# 求X与Y的差集
def X_Y(X, Y):
    D = []
    for x in X:
        if x not in Y:
            D.append(x)
    return D


# Eq_2 估计节点u局部影响力
def LFV(G, u):
    """
    N_u:表示节点u的一跳邻居集
    N_v:表示节点v的一跳邻居集
    """
    n = 1
    N_u = []
    for g in G.out_edges(u):
        if g[1] != u:
            N_u.append(g[1])
    for v in N_u:
        n_v = 0
        N_v = []
        for g in G.out_edges(v):
            if g[1] != u and g[1] != v:
                N_v.append(g[1])
        for s in N_v:
            n_v += (1 / G.in_degree(s))
        n = n + (1 / G.in_degree(v)) + (1 / G.in_degree(v)) * n_v
    return n


# PS(u): 节点集S 激活u 的概率
def PS(G, S, u):
    """
    Pe: 表示u的前一个节点集
    S_n_Pe: Pe与S的交集
    """
    S_n_Pe = []
    Pe = [v[0] for v in G.in_edges(u)]
    # print("Pe", Pe)
    for i in range(len(Pe)):
        if Pe[i] in S:
            S_n_Pe.append(Pe[i])
    m = 1
    for v in S_n_Pe:
        m = m * (1 - 1 / G.in_degree(u))
    return 1 - m


# 公式f1: 节点集S 对其一跳节点的影响效益
def F1(G, S):
    """
    N_s1: 表示节点集S的一跳邻居节点集
    """
    N_s1 = []
    f1 = 0
    for s in S:
        for s_node in G.out_edges(s):
            if s_node[1] not in S and s_node[1] not in N_s1:
                N_s1.append(s_node[1])
    # print("N_s1", N_s1)
    for u in N_s1:
        f1 += PS(G, S, u) * LFV(G, u)
    return f1


# 公式f2: 计算集合S 两跳的值
def F2(G, S):
    """
     N_s1: 表示节点集S的一跳邻居节点集
     N_s2: 表示节点集S的两跳邻居节点集
     N: 表示N_s2-N_s1-S
    """
    N_s1 = []
    for s in S:
        for s_node in G.out_edges(s):
            if s_node[1] not in N_s1:
                N_s1.append(s_node[1])
    # print("N_s1", N_s1)
    N_s2 = []
    for s in N_s1:
        for s_node in G.out_edges(s):
            if s_node[1] not in N_s2:
                N_s2.append(s_node[1])
    # print("N_s2", N_s2)
    N = X_Y(X_Y(N_s2, N_s1), S)
    # print("N_s2-N_s1-S", N)
    f2 = 0
    for u in N:
        v0 = 1
        Pe_n_N_s1 = []
        Pe = [v[0] for v in G.in_edges(u)]
        for i in range(len(Pe)):
            if Pe[i] in N_s1:
                Pe_n_N_s1.append(Pe[i])
        # print("Pe_n_N_s1", Pe_n_N_s1)
        for v in Pe_n_N_s1:
            v0 = v0 * (1 - (1 / G.in_degree(u)) * PS(G, S, v))
        f2 = f2 + (1 - v0) * LFV(G, u)
    return f2


# 适应度值
def EDIV(G, S):
    return a * len(S) + b * F1(G, S) + c * F2(G, S)


# Algorithm 1 计算LID 局部影响力递减
def Local_Influence_Descending(G, k):
    N = []  # 每个节点的影响相对较大的节点集
    V = []  # 计算每个节点的LFV值
    for i in range(1, len(G.nodes()) + 1):
        V.append((i, LFV(G, i)))
    V = sorted(V, key=lambda x: x[1], reverse=True)
    for i in range(1, k + 1):
        up_bound = up_a * i + up_b
        if up_bound > len(V):
            up_bound = len(V)
        node = V[random.randint(0, up_bound - 1)][0]
        N.append(node)
    return N


# Algorithm 3 初始化种群
def Initialization(G, pop, k):
    X = [[0 for j in range(k + 1)] for i in range(pop + 1)]  # 初始化X
    V = []  # 计算每个节点的LFV值
    for i in range(1, len(G.nodes()) + 1):
        V.append((i, LFV(G, i)))
    nodelist = sorted(V, key=lambda x: x[1], reverse=True)
    for i in range(1, pop + 1):
        for j in range(1, k + 1):
            index = j + i * k  # 根据这个公式，因此选择节点从1开始
            if index > len(nodelist):  # 越界判断
                index = index % len(nodelist)
            X[i][j] = nodelist[index - 1][0]
    return X


# 判断X中的元素是否相同
def check_same(X):
    a1 = X[0]
    for i in range(1, len(X)):
        if a1 != X[i]:
            return False
    return True


def uniq(X):
    return X[random.randint(0, len(X) - 1)]


# Algorithm 4 进行差异突变
def Differential_Mutation(G, X, F, pop, k):
    """
    M: 表示变异种群
    X_r1: 基线个体
    X_r2: 差异个体
    X_r3: 差异个体
    """
    # if check_same(X[1:]):
    #     print("种群基因一致，无需突变")
    #     print("种子集合：", X[1][1:])
    #     exit()
    M = [[0] for i in range(pop + 1)]
    X_r1 = X_r2 = X_r3 = []
    for i in range(1, pop + 1):
        X_r1 = X[random.randint(1, pop)][1:]
        if check_same(X[i][1:]):
            X_r2 = Local_Influence_Descending(G, k)
            X_r3 = Local_Influence_Descending(G, k)
        else:
            while X_r2 == X_r3:
                X_r2 = X[random.randint(1, pop)][1:]
                X_r3 = X[random.randint(1, pop)][1:]
        M[i] = X_r1
        D = X_Y(X_r2, X_r3)
        N = F * len(D)
        # print(M)
        # exit()
        for a in range(int(N)):
            V = [LFV(G, x) for x in M[i]]
            j = V.index(min(V))
            if M[i] == D:
                un = uniq(Local_Influence_Descending(G, k))
                while un in M[i]:
                    un = uniq(Local_Influence_Descending(G, k))
                M[i][j] = un
                # print("#", M[i][j])
            else:
                M[i][j] = D[random.randint(0, len(D) - 1)]
                # print("#", M[i][j])
        M[i] = [0] + M[i]
    return M


# Algorithm 5 进行交叉
def Crossover(G, X, M, cr, pop, k):
    """
    :param X: 上一代种群
    :param M: 变异种群
    :param cr: 交叉概率
    :param pop: 种群大小
    :param k: 种子节点数量
    :return: 交叉种群
    """
    C = [[0 for j in range(k + 1)] for i in range(pop + 1)]
    # print("X", len(X), len(X[1]), X)
    # print("M", len(M), len(M[1]), M)
    # print("C", len(C), len(C[1]), C)
    for i in range(1, pop + 1):
        for j in range(1, k + 1):
            r = random.random()
            if r < cr and M[i][j] not in C[i][1:]:
                C[i][j] = M[i][j]
            if r < cr and M[i][j] in C[i][1:] and X[i][j] in C[i][1:]:
                un = 0
                while un in C[i]:
                    un = uniq(Local_Influence_Descending(G, k))
                C[i][j] = un
            if r < cr and M[i][j] in C[i][1:] and X[i][j] not in C[i][1:]:
                C[i][j] = X[i][j]
            if r >= cr and X[i][j] not in C[i][1:]:
                C[i][j] = X[i][j]
            if r >= cr and M[i][j] in C[i][1:] and X[i][j] in C[i][1:]:
                un = 0
                while un in C[i]:
                    un = uniq(Local_Influence_Descending(G, k))
                C[i][j] = un
            if r >= cr and X[i][j] in C[i][1:] and M[i][j] not in C[i][1:]:
                C[i][j] = M[i][j]
    return C


# Algorithm 6 进行选择
def Selection(G, X, C, pop):
    """
    :param G: 图
    :param X: 上一代种群
    :param C: 交叉种群
    :param pop: 种群大小
    :return: 更新后的种群
    """
    for i in range(1, pop + 1):
        if EDIV(G, X[i][1:]) < EDIV(G, C[i][1:]):
            X[i] = C[i]
    return X


# Algorithm 2 DE算法
def LIDDE(G, k):
    S = [0]
    X = Initialization(G, pop, k)
    t = 1
    while t <= g_max:
        # print("——————第{}次迭代——————".format(t))
        # print("上一代的种群：", X[1:])
        M = Differential_Mutation(G, X, F, pop, k)
        C = Crossover(G, X, M, cr, pop, k)
        X = Selection(G, X, C, pop)
        # print("变异的种群：", M[1:])
        # print("交叉的种群：", C[1:])
        # print("选择的种群：", X[1:])
        ED_IV = []
        for x in X[1:]:
            ED_IV.append(EDIV(G, x[1:]))
        # print("EDIV:", sorted(ED_IV, reverse=True))
        t += 1
    for i in range(1, len(X)):
        S.append(EDIV(G, X[i][1:]))
    # print("S", S)
    # print(max(S))
    S = X[S.index(max(S))][1:]
    return S


# 导入数据
def import_graph(path):
    datasets = []
    f = open(path)
    data = f.read()
    rows = data.split('\n')
    for row in rows:
        split_row = row.split('	')
        name = [int(split_row[0]), int(split_row[1])]
        datasets.append(name)
    # print(datasets)
    for i in range(1, len(datasets)):
        datasets[i][0] += 1
        datasets[i][1] += 1
    # print(datasets)
    G = nx.DiGraph()
    G.add_edges_from(datasets[1:])
    pos = nx.shell_layout(G)
    nx.draw(G, pos, node_size=300, with_labels=True, node_color='blue')
    # plt.show()
    return G


if __name__ == "__main__":
    G = example()
    # print(LFV(G, 1))
    # print(F2(G, [2, 3]))
    # print(EDIV(G, [1, 2]))
    # X = Initialization(G, pop, k)
    # print(PS(G, [1, 2, 3, 5], 6))
    # print("种子集合：", LIDDE(G, k))
    G = import_graph("D:/python_env_pytorch_pape/net_work/data/karate.txt")
    S = [i - 1 for i in LIDDE(G, k)]
    print("种子集合：", S)
