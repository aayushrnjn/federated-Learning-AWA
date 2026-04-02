import numpy as np
import torch
import torch.nn.functional as F
import math
import torch.optim as optim
import torch.nn as nn
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import init_model
import math
from copy import deepcopy
import warnings
import torch
from torch.nn import Module
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics.pairwise import cosine_similarity
import datetime
from sklearn.cluster import KMeans
##############################################################################
# General server function
##############################################################################

def receive_client_models(args, client_nodes, select_list, size_weights):
    client_params = []
    for idx in select_list:
        if ('fedlaw' in args.server_method) or ('fedawa' in args.server_method):
            client_params.append(client_nodes[idx].model.get_param(clone = True))
            
        else:
            client_params.append(copy.deepcopy(client_nodes[idx].model.state_dict()))
    
    agg_weights = [size_weights[idx] for idx in select_list]
    agg_weights = [w/sum(agg_weights) for w in agg_weights]

    return agg_weights, client_params



def receive_client_models_pool(args, client_nodes, select_list, size_weights):
    client_params = []
    for idx in select_list:
        if ('fedlaw' in args.server_method) or ('fedawa' in args.server_method):
            client_params.append(client_nodes[idx].model.get_param(clone = True))
         
        else:
            client_params.append(copy.deepcopy(client_nodes[idx].model.state_dict()))
    
    agg_weights = [size_weights[idx] for idx in select_list]

    return agg_weights, client_params

def get_model_updates(client_params, prev_para):
    prev_param = copy.deepcopy(prev_para)
    client_updates = []
    for param in client_params:
        client_updates.append(param.sub(prev_param))
    return client_updates

def get_client_params_with_serverlr(server_lr, prev_param, client_updates):
    client_params = []
    with torch.no_grad():
        for update in client_updates:
            param = prev_param.add(update*server_lr)
            client_params.append(param)
    return client_params



global_T_weights_dict={}

def Server_update(args, central_node, client_nodes, select_list, size_weights,rounds_num=None,change=0):
    '''
    server update functions for baselines
    '''
    global size_weights_global
    global global_T_weights
    if rounds_num==change:
        size_weights_global=size_weights
    

    # receive the local models from clients
    if args.server_method == 'fedawa':
        agg_weights, client_params = receive_client_models_pool(args, client_nodes, select_list, size_weights_global)
    else:
        agg_weights, client_params = receive_client_models(args, client_nodes, select_list, size_weights)
    print(agg_weights)
    

    if args.server_method == 'fedavg':
        avg_global_param = fedavg(client_params, agg_weights)
        
        central_node.model.load_state_dict(avg_global_param)
      
  

    elif args.server_method == 'fedawa':
        # print(rounds_num)

        if rounds_num==change:       
            global_T_weights=torch.tensor(agg_weights, dtype=torch.float32).to('cuda')

        
        avg_global_param,cur_global_T_weight = fedawa(args,client_params, agg_weights,central_node,rounds_num,global_T_weights)
        global_T_weights=cur_global_T_weight
        for i in range(len(select_list)):
            size_weights_global[select_list[i]] = global_T_weights[i]
        print("Global size weights:",size_weights_global)
        central_node.model.load_param(avg_global_param)
  



    else:
        raise ValueError('Undefined server method...')

    return central_node

#fedmy sample



# FedAvg
def fedavg(parameters, list_nums_local_data):
    fedavg_global_params = copy.deepcopy(parameters[0])
    # d=[]
    for name_param in parameters[0]:
        list_values_param = []
        for dict_local_params, num_local_data in zip(parameters, list_nums_local_data):
            # print(dict_local_params[name_param])
            list_values_param.append(dict_local_params[name_param] * num_local_data)
        # print("list_values_param:",list_values_param)
        value_global_param = sum(list_values_param) / sum(list_nums_local_data)
        # print("value_global_param:",value_global_param)
   
        # print("name_param:"+name_param+':',fedavg_global_params[name_param]-value_global_param)


        # print("name_param:"+name_param+':',torch.mean(torch.abs(fedavg_global_params[name_param]-value_global_param)))
        # if name_param[-6:]=="weight":
        # a=1-torch.mean(torch.abs(fedavg_global_params[name_param]-value_global_param))
        # d.append(a.item())
        # d=0.999
        fedavg_global_params[name_param] = value_global_param
    # exit()
    # print(d)
    return fedavg_global_params









def unflatten_weight(M, flat_w):
 
    ws = (t.view(s) for (t, s) in zip(flat_w.split(M._weights_numels), M._weights_shapes))
    
    for (m, n), w in zip(M._weights_module_names, ws):
        # print(type(m))
        # exit()
        # print(m,n,w)
        if 'Batch' in str(type(m)):
            print(m,n,w)
        setattr(m, n, w)
    # exit()
    # yield
    # for m, n in M._weights_module_names:
    #     setattr(m, n, None)




def to_var(x, requires_grad=True):
    if isinstance(x, dict):
        return {k: to_var(v, requires_grad) for k, v in x.items()}
    elif torch.is_tensor(x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, requires_grad=requires_grad)
    else:
        return x

def _cost_matrix(x, y, dis, p=2):
        d_cosine = nn.CosineSimilarity(dim=-1, eps=1e-8)
    
        
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        if dis == 'cos':
            # print('cos_dis')
            C = 1-d_cosine(x_col, y_lin)
        elif dis == 'euc':
            # print('euc_dis')
            C= torch.mean((torch.abs(x_col - y_lin)) ** p, -1)
        return C

def _layerwise_cost_matrix(global_flat_w, flat_w_list, model, head_distance, body_distance, head_weight, p=2):
    """Compute cost matrix using separate distance metrics for head (Linear) and body layers.

    Head (Linear/classifier) weights tend to diverge more across clients under non-IID
    data, so cosine similarity better captures directional drift. Body (Conv/BN) weights
    stay closer and benefit from L2, which is sensitive to small absolute differences.
    The two cost matrices are combined via a convex combination controlled by head_weight.
    """
    global_head = model.get_head_weights(global_flat_w)   # [H]
    global_body = model.get_body_weights(global_flat_w)   # [B]

    client_heads = torch.stack([model.get_head_weights(fw) for fw in flat_w_list])   # [K, H]
    client_bodies = torch.stack([model.get_body_weights(fw) for fw in flat_w_list])  # [K, B]

    C_head = _cost_matrix(global_head.unsqueeze(0), client_heads, head_distance, p)   # [1, K]
    C_body = _cost_matrix(global_body.unsqueeze(0), client_bodies, body_distance, p)  # [1, K]

    return head_weight * C_head + (1.0 - head_weight) * C_body
#fedgroupavg_para group mean
def fedawa(args,parameters, list_nums_local_data,central_node,rounds,global_T_weight):
    param=central_node.model.get_param()

    global_params = copy.deepcopy(param)
   

    
    flat_w_list = [dict_local_params['flat_w'] for dict_local_params in parameters]
    
    

    local_param_list = torch.stack(flat_w_list)
    
    T_weights = to_var(global_T_weight)
    
    
    if args.server_optimizer=='sgd':
        Attoptimizer = torch.optim.SGD([T_weights], lr=0.01, momentum=0.9, weight_decay=5e-4)
    elif args.server_optimizer=='adam':
        Attoptimizer = optim.Adam([T_weights], lr=0.001, betas=(0.5, 0.999))
    
    
    print("T_weights_before update:",torch.nn.functional.softmax(T_weights, dim=0))
   

  


    #num of server update
    
    for i in range(args.server_epochs):
        print("server weight update:",i)
        


        probability_train = torch.nn.functional.softmax(T_weights, dim=0)
        

        if args.layerwise_distance:
            C = _layerwise_cost_matrix(
                global_params['flat_w'].detach(),
                [fw.detach() for fw in flat_w_list],
                central_node.model,
                args.head_distance,
                args.body_distance,
                args.head_weight,
            )
        else:
            C = _cost_matrix(global_params['flat_w'].detach().unsqueeze(0), local_param_list.detach(), args.reg_distance)
     
        reg_loss = torch.sum(probability_train* C, dim=(-2, -1))
        print("reg_loss:",reg_loss)






        client_grad=local_param_list-global_params['flat_w']

    
        column_sum=torch.matmul(probability_train.unsqueeze(0),client_grad) #weighted sum
       

        # cosine sim
        # cos_sim = torch.nn.functional.cosine_similarity(client_grad.unsqueeze(0), column_sum.unsqueeze(1), dim=2)
        # print(cos_sim)
        #
        l2_distance = torch.norm(client_grad.unsqueeze(0) - column_sum.unsqueeze(1), p=2, dim=2)
        
        
        # cosine sim
        # print("Cos_sim:",cos_sim)
        # sim_loss=-(torch.sum(probability_train*cos_sim, dim=(-2, -1)))
        # 
        print("L2_distance:",l2_distance)
        sim_loss=(torch.sum(probability_train*l2_distance, dim=(-2, -1)))

        print("Sim_loss:",sim_loss)
     
        Loss=sim_loss+reg_loss
        Attoptimizer.zero_grad()
        Loss.backward()
        Attoptimizer.step()
        print("step "+str(i)+" Loss:"+str(Loss))


 
    global_T_weight=T_weights.data
    

    print("T_weights_after update:",global_T_weight)

    print("probability_train_after update:",probability_train)



    fedavg_global_params = copy.deepcopy(parameters[0])
    # d=[]

    for name_param in parameters[0]:
        list_values_param = []
        for dict_local_params, num_local_data in zip(parameters, probability_train):
            # print(dict_local_params[name_param])
            list_values_param.append(dict_local_params[name_param] * num_local_data * args.gamma)
        # print("list_values_param:",list_values_param)
        value_global_param = sum(list_values_param) / sum(probability_train)
      
        fedavg_global_params[name_param] = value_global_param
    
    return fedavg_global_params,global_T_weight





