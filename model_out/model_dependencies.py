
from graphviz import Digraph
import pydot



dot_object = pydot.Dot(graph_name="main_graph",rankdir="LR", labelloc='b', 
                       labeljust='r', ranksep=1, dpi=300)
dot_object.set_node_defaults(shape='circle', fixedsize='true',
                             height=.85, width=.85, fontsize=24)
#hyperparametres
node_h0 = pydot.Node(name='h0', texlbl=r'\lambda_{0}', label='<&#955;<SUB>0</SUB>>')
node_h1 = pydot.Node(name='h1', texlbl=r'\lambda_{1}', label='<&#955;<SUB>1</SUB>>')
node_h2 = pydot.Node(name='h2', texlbl=r'\lambda_{2}', label='<&#955;<SUB>2</SUB>>')

#dot_object.add_node(node_h0)
#dot_object.add_node(node_h1)
#dot_object.add_node(node_h2)

# K plate
plate_K = pydot.Cluster(graph_name='plate_k', label='G', fontsize=24)

node_alpha = pydot.Node(name='alpha', texlbl=r'\alpha', label="<&#945;<SUB>g</SUB>>")
node_beta = pydot.Node(name='beta', texlbl=r'\beta', label='<&#946;<SUB>g</SUB>>')
node_eta = pydot.Node(name='eta', texlbl=r'\eta', label='<&#951;<SUB>g</SUB>>')


plate_K.add_node(node_beta)
plate_K.add_node(node_alpha)
plate_K.add_node(node_eta)


# add plate k to graph
dot_object.add_subgraph(plate_K)

# M plate
plate_M = pydot.Cluster(graph_name='plate_M', label='P', fontsize=24)
node_theta = pydot.Node(name='theta', texlbl=r'\theta',
                        label='<&#952;<SUB>p</SUB>>')
node_gamma = pydot.Node(name='gamma', texlbl=r'\gamma', 
                        label='<&#947;<SUB>p</SUB>>')
node_eta_p = pydot.Node(name='etap', texlbl=r'\eta', 
                        label='<&#957;<SUB>p</SUB>>')

plate_M.add_node(node_theta)
plate_M.add_node(node_gamma)
plate_M.add_node(node_eta_p)

# N plate
#node_z = pydot.Node(name='z', texlbl='S_{p}', label='<S<SUB>p</SUB>>')
#plate_M.add_node(node_z)
node_w = pydot.Node(name='w', texlbl='R_{p}', label='<R<SUB>p</SUB>>', 
                    style='filled', fillcolor='lightgray')


plate_M.add_node(node_w)

plate_K.add_subgraph(plate_M)
dot_object.add_subgraph(plate_M)

# Add the edges
#dot_object.add_edge(pydot.Edge(node_h0,node_alpha))
#dot_object.add_edge(pydot.Edge(node_h1,node_beta))
#dot_object.add_edge(pydot.Edge(node_h2,node_eta))
#dot_object.add_edge(pydot.Edge(node_z, node_w))

dot_object.add_edge(pydot.Edge(node_alpha, node_theta))
dot_object.add_edge(pydot.Edge(node_theta, node_w))

dot_object.add_edge(pydot.Edge(node_eta, node_eta_p))
dot_object.add_edge(pydot.Edge(node_gamma, node_w))
dot_object.add_edge(pydot.Edge(node_eta_p, node_w))

dot_object.add_edge(pydot.Edge(node_beta,node_theta))

#dot_object.render('test-output/round-table.gv', view=True) 
dot_object.write_png('lda_graph.png', prog='dot')
