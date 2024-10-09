import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import math
import numpy as np
from tqdm import tqdm


def get_input(file_name):
    df = pd.read_excel(file_name + '.xlsx')
    Zf_name = df['Component'].tolist()
    Strategy_scores = df['Strategy value'].tolist()
    Initial_ms = df['Market share in 2013'].tolist()
    return Zf_name, Strategy_scores, Initial_ms


def f_strategy(Strategy_scores):
    Dic_strategy = {}
    for x in range(len(Strategy_scores)):
        Dic_strategy[zf_name[x]] = Strategy_scores[x]
    return Dic_strategy


def g_add_nodes_1(file_name, G, data={}):
    dic_map = {}
    df1 = pd.read_excel(file_name + '.xlsx')
    _name = df1[file_name].tolist()
    _x = df1['x'].tolist()
    _y = df1['y'].tolist()
    _pos = [(x, y) for x, y in zip(_x, _y)]
    _name_new = [a + '\n' + str(round(data.loc[a, 'Market share'], 5)) for a in _name]
    dic_map = dict(zip(_name, _name_new))
    dic_map_map = dict(zip(_name_new, _name))
    dic_pos = dict(zip(_name_new, _pos))
    G.add_nodes_from(_name_new)
    return_name = _name_new

    return return_name, dic_pos, dic_map, dic_map_map


def g_add_nodes_2(file_name, G):
    dic_map = {}
    df1 = pd.read_excel(file_name + '.xlsx')
    _name = df1[file_name].tolist()
    _x = df1['x'].tolist()
    _y = df1['y'].tolist()
    _pos = [(x, y) for x, y in zip(_x, _y)]
    dic_pos = dict(zip(_name, _pos))
    G.add_nodes_from(_name)
    return_name = _name

    return return_name, dic_pos


def create_G1(nodes_num, connecting_strength):  # Construct the group game network and display the image. The program continues after the image is closed.
    G1 = nx.barabasi_albert_graph(nodes_num, connecting_strength)  # The two parameters correspond to the number of nodes and the connection strength of the Barabási-Albert (BA) scale-free network.
    for i in range(nodes_num):
        G1.add_node(i, Vehicle=np.random.choice(zf_name[0:], p=initial_ms[0:]))
    nx.draw(G1, pos=nx.spring_layout(G1), node_color='#6A5ACD', with_labels=False, alpha=0.5, node_size=100)
    plt.show()
    return G1


def f_individual_strategy(G1, dic_strategy):
    dic_node = dict(G1._node)
    dic_individual_strategy = {}
    for i in range(nodes_num):
        dic_individual_strategy[i] = dic_strategy[dic_node[i]['Vehicle']]
    return dic_individual_strategy


def f_earning(dic_individual_strategy):
    dic_payoff = {}
    for i in dic_individual_strategy:
        dic_payoff[i] = dic_individual_strategy[i] * 0.25
    return dic_payoff


def probability(dic_payoff, num):  # A control function can be established based on the characteristics of different systems to adjust selection probabilities.
    sum_fitness = 0
    for j in range(nodes_num):
        sum_fitness += dic_payoff[j]
    min1 = min(dic_payoff)
    max1 = max(dic_payoff)
    dic_payoff_ = {}
    for i in range(len(dic_payoff)):
        dic_payoff_[i] = (dic_payoff[i] - min1) / (max1 - min1)
    p = []
    k = 0.15
    if num <= 0.5 * epoch:
        a = 1
        k = k * (0.5 * epoch + 1 - num)
    if num > 0.5 * epoch:
        a = -1
        k = k * (num - 0.5 * epoch)
    k = 2 / (0.5 * epoch - 1) * (num - 1) + 1
    for j in range(nodes_num):
        zb = dic_payoff_[j] / sum_fitness
        sigmoid = a * k * (8 * dic_payoff_[j] - 4)  # The sigmoid function can be used to simulate the market vitality of new products or technologies over time, given the presence of fixed substitute products.
        pick = zb * (1 / (1 + math.e ** (sigmoid)))
        p.append(pick)
    p = np.array(p)
    p /= p.sum()
    return p


def calculate_ms(data):
    counters = [0] * len(zf_name)
    for i in range(nodes_num):
        for j in range(len(zf_name)):
            if G1.nodes[i]['Vehicle'] == zf_name[j]:
                counters[j] += 1

    for i in range(len(zf_name)):
        data.iloc[i] = counters[i] / nodes_num

    ms_list1.append(round(counters[0] / nodes_num, 5))
    ms_list2.append(round(counters[1] / nodes_num, 5))

    return data, ms_list1, ms_list2


def ms_line(ms_list1, ms_list2, filename):
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.figure(figsize=(20, 4))

    plt.title('The Evolutionary Trend of the Scenario')
    plt.xlabel('Iterations')
    plt.ylabel('Market share')
    plt.ylim(0, 1)

    epoch = len(ms_list1)
    plt.plot([x for x in range(1, epoch + 1)], ms_list1, label=zf_name[0])
    plt.plot([x for x in range(1, epoch + 1)], ms_list2, label=zf_name[1])
    plt.legend()

    plt.savefig(filename + '.png', dpi=500)
    plt.show()


# Set parameters
zf_name, strategy_scores, initial_ms = get_input('Element strategy values and initial market share')
data = pd.read_excel('Market share of component' + '.xlsx', index_col=0)

num_test = 1  # [1,15]  # This parameter indicates the experiment number. Since the Group State Update involves probabilistic selection, multiple experiments are needed to calculate the average values for improved accuracy.
nodes_num = 600  # {200,400,600,800,1000}  # The number of nodes in the BA scale-free network
connecting_strength = 15  # {9,11,13,15,17,19}  # The connection strength of the BA scale-free network
epoch = 8000  # The number of iterations for each evolution
print_show = 1  # Whether to display the evolution progress bar
print_step = 1000  # Interval for displaying the evolution progress bar
photo_show = 1  # Whether to display the element evolution network image
photo_step = 5  # Interval for displaying the element evolution network image

pd.options.display.float_format = '{:.5f}'.format
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']
matplotlib.rcParams['axes.unicode_minus'] = False

G1 = create_G1(nodes_num, connecting_strength)
dic_strategy = f_strategy(strategy_scores)
dic_individual_strategy = f_individual_strategy(G1, dic_strategy)
dic_payoff = f_earning(dic_individual_strategy)

ms_list1 = []
ms_list2 = []

for i in tqdm(range(1, epoch + 1)):
    if print_show == 1:
        if i % print_step == 0:
            print()
            print('-------------------------EPOCH  ' + str(i) + '-------------------------')

    if photo_show == 1:
        plt.ion()
        if i % photo_step == 0:
            G = nx.Graph()
            component_name_new, dic_pos1, dic_map, dic_map_map = g_add_nodes_1('Component', G, data)
            technology_name, dic_pos2 = g_add_nodes_2('Technology', G)
            need_name, dic_pos3 = g_add_nodes_2('Demand', G)

            df_edge1 = pd.read_excel('Component to component.xlsx', sheet_name='1')
            lst_resource1 = df_edge1['Resource'].tolist()
            lst_target1 = df_edge1['Goal'].tolist()
            lst_weight1 = df_edge1['Weight'].tolist()
            for resource1, target1, weight in zip(lst_resource1, lst_target1, lst_weight1):
                G.add_edge(dic_map[resource1], dic_map[target1], weight=weight)

            df_edge2 = pd.read_excel('Technology to component.xlsx')
            lst_resource2 = df_edge2['Resource'].tolist()
            lst_target2 = df_edge2['Goal'].tolist()
            for resource2, target2 in zip(lst_resource2, lst_target2):
                G.add_edge(resource2, dic_map[target2])

            df_edge3 = pd.read_excel('Demand to component.xlsx')
            lst_resource3 = df_edge3['Resource'].tolist()
            lst_target3 = df_edge3['Goal'].tolist()
            for resource3, target3 in zip(lst_resource3, lst_target3):
                G.add_edge(resource3, dic_map[target3])

            pos_all = {**dic_pos1, **dic_pos2, **dic_pos3}

            node_color = []
            norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
            mapper = cm.ScalarMappable(norm=norm, cmap=cm.PuRd)

            for node in G:
                if node in component_name_new:
                    weight = round(data.loc[dic_map_map[node], 'Market share'], 5)
                    node_color.append(mapper.to_rgba(weight))
                elif node in technology_name:
                    node_color.append('wheat')
                elif node in need_name:
                    node_color.append('pink')

            nx.draw_networkx(G, pos=pos_all, node_size=1100, node_color=node_color, edge_color='silver', font_size=10)

            plt.tight_layout()
            plt.pause(0.1)
            plt.cla()

    selected_cxz = np.random.choice([x for x in range(nodes_num)], p=probability(dic_payoff, i))
    dic_nbr = G1[selected_cxz]
    nbr_index_list = [index for index in dic_nbr]
    update_node_index = np.random.choice(nbr_index_list)
    G1.nodes[update_node_index].update({'Vehicle': G1.nodes[selected_cxz]['Vehicle']})
    dic_individual_strategy[update_node_index] = dic_individual_strategy[selected_cxz]
    dic_payoff[update_node_index] = dic_payoff[selected_cxz]
    calculate_ms(data)

# Output results
ms_line(ms_list1, ms_list2, str(nodes_num) + "_" + str(connecting_strength) + "_" + str(num_test) + "_" + 'Evolutionary trend')

list_all = np.stack((ms_list1, ms_list2), axis=0).tolist()
list_out = pd.DataFrame(list_all, index=zf_name, columns=range(1, epoch + 1))
print(list_out)
list_out.to_excel(str(nodes_num) + "_" + str(connecting_strength) + "_" + str(num_test) + "_" + 'Evolutionary state value' + '.xlsx')
