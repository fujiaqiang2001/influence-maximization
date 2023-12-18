import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time
from igraph import *
import pandas as pd
import openpyxl


# 导入数据
def import_graph(path):
    datasets = []
    f = open(path)
    data = f.read()
    rows = data.split('\n')
    for row in rows:
        split_row = row.split('	')
        name = (int(split_row[0]), int(split_row[1]))
        # print(name)
        # exit()
        datasets.append(name)
    print(datasets)

    g = Graph(directed=True)
    g.add_vertices(range(datasets[0][0]))
    g.add_edges(datasets[1:])

    return g


# 简单的例子
def example():
    source = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5]
    target = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9, 6, 7, 8, 9]
    g = Graph(directed=True)
    g.add_vertices(range(10))
    g.add_edges(zip(source, target))
    print(g)
    G = nx.DiGraph()
    G.add_edges_from(zip(source, target))
    pos = nx.shell_layout(G)
    nx.draw(G, pos, node_size=300, with_labels=True, node_color='blue')
    plt.show()
    return g


# IC模型计算节点影响力值
def IC(g, S, p=0.01, mc=1000):
    """
    Input:  graph object, set of seed nodes, propagation probability
            and the number of Monte-Carlo simulations
    Output: average number of nodes influenced by the seed nodes
    """

    # Monte-Carlo模拟循环
    spread = []
    for i in range(mc):

        # 模拟传播过程
        # new_active：新激活节点, A：激活的节点
        # 初始化两个节点都为种子节点
        new_active, A = S[:], S[:]
        while new_active:

            # new_ones：邻居节点
            new_ones = []
            for node in new_active:
                # 确定激活结点，seed固定随机种子,确定每次i产生的随机数一样
                np.random.seed(i)
                # 从均匀分布中抽取样本
                success = np.random.uniform(0, 1, len(g.neighbors(node, mode="out"))) < p
                # 在success = True时，new_ones加入node的邻居结点
                new_ones += list(np.extract(success, g.neighbors(node, mode="out")))

            # 计算新激活的节点
            new_active = list(set(new_ones) - set(A))

            # 将新激活的节点添加到激活的节点集合中
            A += new_active

        # 激活节点的个数加入到spread
        spread.append(len(A))

    # 根据种子结点，通过mc模拟后,计算出影响力的平均值
    return np.mean(spread)


# Greedy算法
def greedy(g, k, p=0.01, mc=1000):
    """
    Input:  graph object, number of seed nodes
    Output: optimal seed set, resulting spread, time for each iteration
    """

    # S:种子节点集合, spread：影响力值集合, timelapse：运行时间集合, LOOKUPS：每个节点的轮次集合
    S, spread, timelapse, LOOKUPS = [], [], [], []
    start_time = time.time()

    # 找出影响力值最大的k个节点
    for _ in range(k):

        # best_spread：最大影响力值 n：轮次
        best_spread = 0
        n = 0
        # 遍历除种子节点以外的所有节点
        for j in set(range(g.vcount())) - set(S):
            n += 1
            # 进行扩散，加入结点j到S中去，计算S的影响力
            s = IC(g, S + [j], p, mc)

            # 更新传播到目前为止，影响力值最大的节点
            if s > best_spread:
                best_spread, node = s, j

        # 将选定的节点添加到种子集
        S.append(node)
        LOOKUPS.append(n)

        # 添加预期影响力值和运行时间
        spread.append(best_spread)
        timelapse.append(time.time() - start_time)

    return S, spread, timelapse, LOOKUPS


# CELF算法
def celf(g, k, p=0.01, mc=1000):
    """
    Input:  graph object, number of seed nodes
    Output: optimal seed set, resulting spread, time for each iteration
    """

    # --------------------
    # 用贪心算法寻找第一个节点
    # --------------------

    start_time = time.time()
    # 计算每个节点的影响力值
    marg_gain = [IC(g, [node], p, mc) for node in range(g.vcount())]

    # 根据节点影响力值，进行降序排列 (节点, 影响力值)
    Q = sorted(zip(range(g.vcount()), marg_gain), key=lambda x: x[1], reverse=True)

    # 选择第一个节点并从候选列表中删除，即最优的第一个种子节点
    # S:种子节点集合, spread:所选节点的影响力值, SPREAD:种子节点影响力值集合
    S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
    Q, LOOKUPS, timelapse = Q[1:], [g.vcount()], [time.time() - start_time]

    # --------------------
    # 使用列表排序过程查找k-1个节点
    # --------------------

    for _ in range(k - 1):

        check, node_lookup = False, 0

        while not check:
            node_lookup += 1

            # 获取顶部节点
            current = Q[0][0]

            # 计算S + [current]与spread影响力值的差值h，重新加入队列
            h = IC(g, S + [current], p, mc) - spread
            Q[0] = (current, h)

            # 重新排序
            Q = sorted(Q, key=lambda x: x[1], reverse=True)

            # 检查顶部节点在排序后是否仍然位于顶部
            check = (Q[0][0] == current)

        # 更新所选种子节点集合的影响力值
        spread = spread + Q[0][1]

        S.append(Q[0][0])
        SPREAD.append(spread)
        LOOKUPS.append(node_lookup)
        timelapse.append(time.time() - start_time)

        # 从列表中删除选定的节点
        Q = Q[1:]

    # S:种子节点集合, spread：影响力值集合, timelapse：运行时间集合, LOOKUPS：每个节点的轮次集合
    return S, SPREAD, timelapse, LOOKUPS


# 计算种子节点的时间图像
def show_graph_time(greedy_output, celf_output):
    plt.plot(range(1, len(greedy_output[2]) + 1), greedy_output[2], marker='*', label="Greedy", color="#FBB4AE")
    plt.plot(range(1, len(celf_output[2]) + 1), celf_output[2], marker='o', label="CELF", color="#B3CDE3")
    plt.ylabel('Computation Time (Seconds)')
    plt.xlabel('Size of Seed Set')
    plt.title('Computation Time')
    plt.legend()
    plt.show()


# 寻找每个种子节点的轮次图像
def show_graph_lookups(greedy_output, celf_output):
    plt.plot(range(1, len(greedy_output[3]) + 1), greedy_output[3], marker='*', label="Greedy", color="#FBB4AE")
    plt.plot(range(1, len(celf_output[3]) + 1), celf_output[3], marker='o', label="CELF", color="#B3CDE3")
    plt.ylabel('count')
    plt.xlabel('Size of Seed Set')
    plt.title('Lookups')
    plt.legend()
    plt.show()


# 预估影响力值的图像
def show_graph_size(greedy_output, celf_output):
    # Plot Expected Spread by Seed Set Size
    plt.plot(range(1, len(greedy_output[1]) + 1), greedy_output[1],
             marker='d', label="Greedy", color="#FBB4AE",
             linewidth=4)
    plt.plot(range(1, len(celf_output[1]) + 1), celf_output[1],
             marker='o', label="CELF", color="#B3CDE3")
    plt.xlabel('Size of Seed Set')
    plt.ylabel('Expected Spread')
    plt.title('Expected Spread')
    plt.legend()
    plt.show()


# 输出结果数据
def excel_write(name, data):
    wb = openpyxl.Workbook()
    wb.save("D:/python_env_pytorch_pape/net_work/data/text/" + name + '.xlsx')
    df = pd.DataFrame(data)
    # 将 DataFrame 写入 Excel 文件
    df.to_excel("D:/python_env_pytorch_pape/net_work/data/text/" + name + '.xlsx', index=False)


# 计算['karate-int.txt', 'netscience-int.txt', 'blog-int.txt', 'email_1133_5451-int.txt']
def total_file(s, file):
    for j in file:
        print("加载文件：", j)

        # 数据地址
        d = s + j
        G = import_graph(path=d)

        s_celf = [33, 32, 16, 25, 7]
        s_LIDDE = [32, 31, 16, 27, 33]
        s_LIDDE_finish = [10, 12, 16, 33, 0]
        s_LIDDE_D = [12, 29, 16, 33, 10]
        print(IC(G, s_celf, p=0.01, mc=1000))
        print(IC(G, s_LIDDE, p=0.01, mc=1000))
        print(IC(G, s_LIDDE_finish, p=0.01, mc=1000))
        print(IC(G, s_LIDDE_D, p=0.01, mc=1000))
        exit()

        # seed = [2]
        seed_k = 5

        greedy_output = greedy(G, seed_k, p=0.01, mc=1000)
        celf_output = celf(G, seed_k, p=0.01, mc=1000)
        # 输出结果

        data = {'greedy_S': greedy_output[0],
                'greedy_spread': greedy_output[1],
                'greedy_timelapse': greedy_output[2],
                'greedy_LOOKUPS': greedy_output[3],
                'celf_S': celf_output[0],
                'celf_spread': celf_output[1],
                'celf_timelapse': celf_output[2],
                'celf_LOOKUPS': celf_output[3]
                }
        # 输出数据
        # excel_write(j, data)
        for i in data:
            print(i, ":", data[i])

        # 绘制图像
        show_graph_time(greedy_output, celf_output)
        show_graph_lookups(greedy_output, celf_output)
        show_graph_size(greedy_output, celf_output)
        plt.title(j)
        plt.show()
        print("本次计算结束")
        exit()


if __name__ == "__main__":
    # G = example()
    # seed_k = 5
    # celf_output = celf(G, seed_k, p=0.2, mc=1000)
    # print(celf_output)

    s = "D:/python_env_pytorch_pape/net_work/data/networks/"
    file = ['karate-int.txt', 'netscience-int.txt', 'blog-int.txt', 'email_1133_5451-int.txt']
    total_file(s, file)


