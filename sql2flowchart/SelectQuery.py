"""
Simple select query: Select query with no subquery
"""

import sqlparse
from sqlparse.sql import Token, TokenList
import re
import json
import random, string
import copy
import numpy as np
import matplotlib.pyplot as plt
# import textwrap
import plotly.graph_objects as go
import networkx as nx
from collections import defaultdict

class SelectQuery():
    """
    Class for parsing select query into dict. Subqueries are assigned dynamically generated names.
    """
    def __init__(self,query_text):
        """
        Arguments:
        query_text: String containing select query
        """
        self.query_text = query_text
        self.keywords = ['select','from','where','group','having']
    
    def parse_select_query(self):
        self.init_parse_query()
        
        # get all initial select queries
        self.get_all_queries()
        # convert all queries to simple queries
        ks = copy.deepcopy(list(self.all_queries.keys()))
        for k in ks:
            if isinstance(self.all_queries[k]['from'],list):
                self.all_queries[k] = self.make_simple(self.all_queries[k])
        # parse statements
        ks = copy.deepcopy(list(self.all_queries.keys()))
        for k in ks:
            # select
            if isinstance(self.all_queries[k]['select'],list):
                self.all_queries[k]['select'] = self.parse_select_statement(self.all_queries[k]['select'])
            
            # where
            self.all_queries[k]['where'] = self.parse_where_have_statement(self.all_queries[k]['where'])
             
            # group
            self.all_queries[k]['group'] = self.parse_groupby_statement(self.all_queries[k])
            
            # having
            self.all_queries[k]['having'] = self.parse_where_have_statement(self.all_queries[k]['having'])
            
    
    def init_parse_query(self):
        parsed_q = sqlparse.parse(self.query_text)[0] # assumed that q has a single query
        q_tokens = parsed_q.tokens
        q_tokens_c = self._clean_query(q_tokens)
        self.init_parse = q_tokens_c

    def get_all_queries(self):
        all_queries={}
        
        if self.init_parse[0].match(sqlparse.tokens.Keyword.CTE,['with']):
            # get all subqueries from with

            # list,dict of all subqueries
            subq_list_raw = self._clean_query(self.init_parse[1].tokens)
            for subq in subq_list_raw:
                all_queries.update(self.parse_simple_select(self._get_select_from_par(subq),subq.get_name()))

            if self.init_parse[2].match(sqlparse.tokens.Keyword.DML,['select']):
                all_queries.update(self.parse_simple_select(self.init_parse[2:],'out'))

        elif self.init_parse[0].match(sqlparse.tokens.Keyword.DML,['select']):
            # simpler query, parse it
            all_queries.update(self.parse_simple_select(self.init_parse,'out'))

        self.all_queries = all_queries
    
    def parse_simple_select(self,q_tokens,q_name):
        if q_tokens:
            assert self._check_token_is_base_keyword(q_tokens[0],'select')
        
            if self._has_union_all(q_tokens):
                # do something
                sub_q_union = self._split_by_union_all(q_tokens)
                op_dict = {}
                q_parse_dict_main = {kw:[] for kw in self.keywords}
                q_parse_dict_main['from'] = {'input_tables':{},'combine_type':'union'}

                for subq in sub_q_union:
                    subq_name = self._generate_name()
                    q_parse_dict_main['from']['input_tables'][subq_name] = subq_name
                    op_dict.update(self.parse_simple_select(subq,subq_name))

                q_parse_dict_main['select'] = {k:k for k,v in self.parse_select_statement(op_dict[subq_name]['select']).items()}
                op_dict[q_name] = q_parse_dict_main
                return op_dict

            else:
                # initial high level parsing
                q_parse_dict_1 = {kw:[] for kw in self.keywords}
                base_keyword_idx = 0
                for t in q_tokens:
                    is_kw_flag = False
                    for kw in self.keywords:
                        if self._check_token_is_base_keyword(t,kw):
                            key = kw
                            is_kw_flag = True
                    if not is_kw_flag:
                        q_parse_dict_1[key].append(t)

                return {q_name:q_parse_dict_1}
    
    def make_simple(self,parsed_t_s):
        if self._is_simple_select(parsed_t_s):
            parsed_t_s['from'] = self.parse_from_statement(parsed_t_s['from'])
            return parsed_t_s
        else:
            new_from = self.parse_from_statement(parsed_t_s['from'])
            for i in range(len(parsed_t_s['from'])):
                t = parsed_t_s['from'][i]
                if isinstance(t,sqlparse.sql.Identifier):
                    subq_name = self._generate_name()
                    sel = self.parse_simple_select(self._get_select_from_par(t),subq_name)
                    if sel:
                        new_from['input_tables'][t.get_name()] = subq_name
                        for k,v in sel.items():
                            if isinstance(v['from'],list):
                                self.all_queries[k] = self.make_simple(v)
                            else:
                                self.all_queries[k] = v
                elif isinstance(t,sqlparse.sql.IdentifierList):
                    for tt in self._clean_query(t.tokens):
                        subq_name = self._generate_name()
                        sel = self.parse_simple_select(self._get_select_from_par(tt),subq_name)
                        if sel:
                            new_from['input_tables'][tt.get_name()] = subq_name
                            for k,v in sel.items():
                                if isinstance(v['from'],list):
                                    self.all_queries[k] = self.make_simple(v)
                                else:
                                    self.all_queries[k] = v
            parsed_t_s['from'] = new_from
            return parsed_t_s              
    
    
    def parse_select_statement(self,s_tokens):
        op_dict = {}
        for t in s_tokens:
            if isinstance(t,sqlparse.sql.Identifier):
#                 if isinstance(self._clean_query(t.tokens)[0],sqlparse.sql.Case):
#                     op_dict[t.get_name()] = self._clean_col_def(t.value,t.get_name())
#                 else:
                op_dict[t.get_name()] = self._clean_col_def(t.value,t.get_name())
            elif isinstance(t,sqlparse.sql.IdentifierList):
                for tt in self._clean_query(t.tokens):
                    op_dict[tt.get_name()] = self._clean_col_def(tt.value,tt.get_name())
            elif t.match(sqlparse.tokens.Wildcard,'\*$',regex=True):
                op_dict[t.value] = t.value
        return op_dict


    def parse_from_statement(self,f_tokens):
        op_dict = {'input_tables':{}}

        # only identifier or identifier list present in f_tokens
        only_identifiers_flag=True
        for t in f_tokens:
            if not (isinstance(t,sqlparse.sql.Identifier) or isinstance(t,sqlparse.sql.IdentifierList)):
                only_identifiers_flag=False

        if only_identifiers_flag:
            for t in f_tokens:
                if isinstance(t,sqlparse.sql.Identifier):
                    op_dict['input_tables'][t.get_name()] = t.get_real_name()
                elif isinstance(t,sqlparse.sql.IdentifierList):
                    for tt in self._clean_query(t.tokens):
                        op_dict['input_tables'][tt.get_name()] = tt.get_real_name()
            return op_dict

        # join statement
        op_dict['joins'] = {}
        op_dict['combine_type'] = 'join'
        for i,t in enumerate(f_tokens):
            if isinstance(t,sqlparse.sql.Identifier):
                op_dict['input_tables'][t.get_name()] = t.get_real_name()

            if self._check_token_is_base_keyword(t,['left join','right join','inner join','join','full outer join']):
                key = f_tokens[i+1].get_name()
                join_dict = {}
                join_dict['type'] = t.value.replace(' join','')
                if join_dict['type']=='join':
                    join_dict['type'] = 'inner'
                join_dict['on'] = f_tokens[i+3].value
                op_dict['joins'][key] = join_dict

        return op_dict        
    
    def parse_where_have_statement(self,w_tokens):
        op_list = []
        for t in w_tokens:
            op_list.append(t.value)
        return ' '.join(op_list)
    
    def parse_groupby_statement(self,q_tokens):
        op_list = []
        g_tokens = q_tokens['group']
        for t in g_tokens:
            if isinstance(t,sqlparse.sql.Identifier):
                op_list.append(t.get_name())
            elif t.match(sqlparse.tokens.Literal.Number.Integer,[str(i) for i in range(1,1001)]):
                op_list.append(list(q_tokens['select'].keys())[int(t.value)-1])
            elif isinstance(t,sqlparse.sql.IdentifierList):
                for tt in self._clean_query(t.tokens):
                    if isinstance(tt,sqlparse.sql.Identifier):
                        op_list.append(tt.get_name())
                    if tt.match(sqlparse.tokens.Literal.Number.Integer,[str(i) for i in range(1,1001)]):
                        op_list.append(list(q_tokens['select'].keys())[int(tt.value)-1])
        return op_list
     
    def plot_query(self,
                   graph_coords_params={'x_incr':5,'width':1,'ratio':0.7},
                   plot_params={'edgewidth':0.5,
                                'edgecolor':'#888',
                                'marker_size':{'raw_table':60,'derived_table':60},
                                'marker_shape':{'raw_table':'square','filter':'triangle-right','aggregate':'triangle-right'},
                                'marker_edgewidth':2,
                                'title':'Query flow',
                                'titlefont_size':16
                               }
                  ):
        self.create_nodes_edges()
        self.create_graph(**graph_coords_params)
        self.plot_graph(**plot_params)
        
        
        
    def create_nodes_edges(self):
        self.nodes = []
        self.edges = []
        self.min_state = {}
        for k,v in self.all_queries.items():
            self.nodes.append(k)
#             for inp in list(v['from']['input_tables'].values()):
#                 self.edges.append((inp,k))
            prev = list(v['from']['input_tables'].values())
            for proc in ['where','group','having']:
                
                if len(v[proc])>0:
                    try:
                        self.min_state[k]
                    except KeyError:
                        self.min_state[k] = k+'_'+proc
                    self.nodes.append(k+'_'+proc)
                    for inp in prev:
                        self.edges.append((inp,k+'_'+proc))
                    prev = [k+'_'+proc]
            try:
                self.min_state[k]
            except KeyError:
                self.min_state[k] = k
            for inp in prev:
                self.edges.append((inp,k))
            self.nodes.extend(list(v['from']['input_tables'].values()))


        self.nodes = list(np.unique(self.nodes))
        
#         for i in range(len(self.edges)):
#             try:
#                 if not (self.edges[i][1]==self.edges[i][0]+'_where' or\
#                         self.edges[i][1]==self.edges[i][0]+'_group' or\
#                         self.edges[i][1]==self.edges[i][0]+'_having'):
#                     self.edges[i] = (self.max_state[self.edges[i][0]],self.edges[i][1])
#             except KeyError:
#                 continue
                
        self.node_properties = {n:{} for n in self.nodes}
        for n in self.nodes:
            self.node_properties[n]['type'] = 'derived_table'
            self.node_properties[n]['text'] = ''
            self.node_properties[n]['pos'] = (0,0)
            try:
                self.node_properties[n]['combine_info']
            except KeyError:
                self.node_properties[n]['combine_info'] = ''
            if n not in self.all_queries.keys():
                self.node_properties[n]['type'] = 'raw_table'
            if re.search('_where$',n):
                self.node_properties[n]['type'] = 'filter'
                self.node_properties[n]['text'] = self.all_queries[re.sub('_where$','',n)]['where']
            elif re.search('_having$',n):
                self.node_properties[n]['type'] = 'filter'
                self.node_properties[n]['text'] = self.all_queries[re.sub('_having$','',n)]['having']
            elif re.search('_group$',n):
                self.node_properties[n]['type'] = 'aggregate'
                self.node_properties[n]['text'] = 'by '+','.join(self.all_queries[re.sub('_group$','',n)]['group'])
            elif self.node_properties[n]['type'] == 'derived_table':
                self.node_properties[n]['text'] = '<br><b>columns:</b> '+json.dumps(self.all_queries[n]['select'],indent=2,separators=(",<br>","=")).replace("{","").replace("}","").replace("\"","")
                
                if len(self.all_queries[n]['from']['input_tables'])>1:
                    min_state = self.min_state[n]
                    self.node_properties[min_state]['combine_info'] = 'combine type: '+self.all_queries[n]['from']['combine_type']
                    if self.all_queries[n]['from']['combine_type']=='join':
                        for k,v in self.all_queries[n]['from']['joins'].items():
                            self.node_properties[min_state]['combine_info']+='<br>'
                            self.node_properties[min_state]['combine_info']+=self.all_queries[n]['from']['input_tables'][k]+'('+k+'):'
                            self.node_properties[min_state]['combine_info']+=v['type']+' join on ' + v['on'] + '<br>'
                

    def create_graph(self,x_incr,width,ratio):
        self.G = nx.DiGraph()

        self.G.add_nodes_from([(n,self.node_properties[n]) for n in self.nodes])
        self.G.add_edges_from(self.edges)
        self.assign_coords_rec(x_incr,width,ratio)
        
#     def create_coords(self,init_coords,incr):
#         curr_level = ['out']
#         curr_x = init_coords[0]
#         while len(curr_level)>0:
#             for n in curr_level:
#                 self.G.nodes[n]['pos'] = (curr_x,self.G.nodes[n]['pos'][1])
#             curr_level = self._get_all_predecessors(curr_level)
#             curr_x -= incr[0]

#         raw_level = []
#         for n in self.G.nodes():
#             if self.G.nodes[n]['type']=='raw_table':
#                 self.G.nodes[n]['pos'] = (curr_x,self.G.nodes[n]['pos'][1])
#                 raw_level.append(n)

#         curr_level = raw_level

#         while len(curr_level)>0:
#             curr_y = init_coords[1]
#             for n in curr_level:
#                 if len(list(self.G.predecessors(n)))>0:
#                     curr_y = np.mean([self.G.nodes[pn]['pos'][1] for pn in self.G.predecessors(n)])
#                 else:
#                     curr_y -= incr[1]
#                 self.G.nodes[n]['pos'] = (self.G.nodes[n]['pos'][0],curr_y)
#             curr_level = self._get_all_successors(curr_level)
    
    def assign_coords_rec(self,x_incr,width,ratio,n='out'):
        if n=='out':
            self.G.nodes[n]['pos'] = (0,0)

        all_c = list(self.G.predecessors(n))
        num_c = len(all_c)
        if num_c==0:
            return True
        elif num_c==1:
            y_incr = np.array([0])
        else:
            y_incr = width*np.linspace(0,1,num_c) - (width/2)

        for i in range(num_c):
            c = all_c[i]
            self.G.nodes[c]['pos'] = (self.G.nodes[n]['pos'][0]-x_incr,self.G.nodes[n]['pos'][1]+y_incr[i])
            self.assign_coords_rec(x_incr,width*ratio,ratio,c)
        return True

    def plot_graph(self,edgewidth,edgecolor,marker_size,marker_shape,marker_edgewidth,title,titlefont_size):
        edge_trace,node_traces = self.configure_graph_plot(edgewidth,edgecolor,marker_size,marker_shape,marker_edgewidth)
        self.fig = go.Figure(data=[edge_trace, *node_traces],
                        layout=go.Layout(
                            title=title,
                            titlefont_size=titlefont_size,
                            showlegend=True,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        self.fig.show()
        
    def configure_graph_plot(self,edgewidth,edgecolor,marker_size,marker_shape,marker_edgewidth):
        edge_x = []
        edge_y = []
        for edge in self.G.edges():
            x0, y0 = self.G.nodes[edge[0]]['pos']
            x1, y1 = self.G.nodes[edge[1]]['pos']
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=edgewidth, color=edgecolor),
            hoverinfo='none',
            mode='lines')

        
        
        marker_size_dict=defaultdict(lambda: 30)
        marker_size_dict.update(marker_size)
        
        marker_shape_dict=defaultdict(lambda: 'circle')
        marker_shape_dict.update(marker_shape)
        node_traces=[]
        for tp in ['raw_table','filter','aggregate','derived_table']:
            node_x = []
            node_y = []
            node_hover_text = []
            node_text = []
            for n in self.G.nodes():
                if self.G.nodes[n]['type']==tp:
                    x, y = self.G.nodes[n]['pos']
                    node_x.append(x)
                    node_y.append(y)
                    
                    nm,txt = self._get_node_texts(n)
                    
                    node_text.append(nm)
                    node_hover_text.append(txt)
                
                
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                name=tp,
                hovertext=node_hover_text,
                hovertemplate='%{hovertext}',
                text=node_text,
                marker=dict(
#                     showscale=False,
                    # colorscale options
                    #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                    #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                    #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
#                     colorscale='YlGnBu',
#                     reversescale=True,
#                     color=[],
                    symbol=marker_shape_dict[tp],
                    size=marker_size_dict[tp],
                    line_width=marker_edgewidth)
            )

#             node_trace.text = node_hover_text
            node_traces.append(node_trace)
            
        return edge_trace,node_traces
    
    # helper functions
    def _clean_col_def(self,c,n):
        return re.sub('\ '+n+'$','',re.sub('\ as\ '+n+'$','',c.split('--')[0].strip())).strip()

    def _generate_name(self,l=8):
        x = ''.join(random.choices(string.ascii_letters + string.digits, k=l))
        return x
    
    def _has_union_all(self,q_t):
        return max([self._check_token_is_base_keyword(t,'union all') for t in q_t])
    
    def _is_simple_select(self,q_t):
        from_t = []
        for t in q_t['from']:
            from_t.extend(list(t.flatten()))
        return not max([self._check_token_is_base_keyword(t,'select') for t in from_t])
        
    def _check_token_is_base_keyword(self,t,kw):
        if 'select' in kw:
            return t.match(sqlparse.tokens.Keyword.DML,kw)
        else:
            return t.match(sqlparse.tokens.Keyword,kw)

    def _get_select_from_par(self,subq_token):
        for t in subq_token.tokens:
            if isinstance(t,sqlparse.sql.Parenthesis):
                return self._clean_query(t.tokens)
        
    def _clean_query(self,tokens):
        tokens_exp = []
        tokens_cleaned = []
        for t in tokens:
            if isinstance(t,sqlparse.sql.Where):
                for tt in t.tokens:
                    tokens_exp.append(tt)
            else:
                tokens_exp.append(t)

        for t in tokens_exp:
            append_flag = True
            for pattern in ['Token.Text.Whitespace*','Token.Punctuation*']:
                if re.match(pattern,str(t.ttype)):
                    append_flag = False
            for t_type in [sqlparse.sql.Comment]:
                if type(t)==t_type:
                    append_flag = False

            if append_flag:
                tokens_cleaned.append(t)

        return tokens_cleaned
    
    def _split_by_union_all(self,q_t):
        op_list = []
        op_sub_list = []
        for t in q_t:
            if self._check_token_is_base_keyword(t,'union all'):
                op_list.append(op_sub_list)
                op_sub_list = []
            else:
                op_sub_list.append(t)
        op_list.append(op_sub_list)
        
        return op_list
    
    def _get_all_successors(self,n_list):
        successors=[]
        for n in n_list:
            successors.extend(self.G.successors(n))
        successors = list(np.unique(successors))
        return successors

    def _get_all_predecessors(self,n_list):
        predecessors=[]
        for n in n_list:
            predecessors.extend(self.G.predecessors(n))
        predecessors = list(np.unique(predecessors))
        return predecessors
    
    def _get_node_texts(self,n):
        nm = n
        for proc in ['where','group','having']:
            if re.search('_'+proc+'$',n):
                nm = re.sub('_'+proc+'$','',n)
        txt = '<b>'+nm+'</b> (' +n+ ') <br><b>'+self.G.nodes[n]['type']+'</b>: '+self.G.nodes[n]['text']+'<br>'#'<br>'.join(textwrap.wrap(self.G.nodes[n]['text'],width=20))+'<br>'
        if len(list(self.G.predecessors(n)))>0:
            txt += '<br><b>Predecessors</b>: '+','.join(list(self.G.predecessors(n))) +'<br>'
            txt += self.G.nodes[n]['combine_info']
        if n!='out':
            txt += '<br><b>Successors</b>: '+','.join(list(self.G.successors(n))) +'<br>'

#         txt = txt+'<br><b>'+self.G.nodes[n]['type']+'</b>: '+'<br>'.join(textwrap.wrap(self.G.nodes[n]['text'],width=10))
        
        return nm,txt
