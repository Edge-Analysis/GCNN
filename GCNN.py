class GCNN:
    def __init__(self,graph_tensor,data_source='graph',target=('type','node_name','feature_name')):
        self.target = target
        if data_source == 'graph':
            self.graph_tensor = graph_tensor
        elif data_source == 'pandas':
            self.graph_tensor = self.from_pandas(graph_tensor)
        elif data_source == 'networkx':
            self.graph_tensor = self.from_networkx(graph_tensor)
        else:
            print("WARNING: Unknown data source")
        
    def from_pandas(self,graph_dict):
        node_sets = {}
        for name, df in graph_dict['node_dfs'].items():
            if len(df.columns) == 0:
                node_sets[name] = tfgnn.NodeSet.from_fields(
                    sizes = [len(df)])
            else:
                features = {}
                for col in df.columns:
                    if df[col].dtype == int:
                        df[col] = df[col].astype('int32')
                    elif df[col].dtype == float:
                        df[col] = df[col].astype('float32')
                    features[col] = df[col].values.reshape(len(df),1)
                node_sets[name] = tfgnn.NodeSet.from_fields(
                    sizes = [len(df)],
                    features = features)

        edge_sets = {}
        for name, df in graph_dict['edge_dfs'].items():
            source = graph_dict['node_dfs'][df[0]]
            idx_name = source.index.name
            source = source.reset_index()[idx_name].reset_index()
            source = pd.merge(df[2]['source'],source,how='left',left_on='source',right_on=idx_name)['index']
            source = source.astype('int32').values

            target = graph_dict['node_dfs'][df[1]]
            idx_name = target.index.name
            target = target.reset_index()[idx_name].reset_index()
            target = pd.merge(df[2]['target'],target,how='left',left_on='target',right_on=idx_name)['index']
            target = target.astype('int32').values

            if len(df[2].columns) == 2:
                edge_sets[name] = tfgnn.EdgeSet.from_fields(
                    sizes = [len(df[2])],
                    adjacency = tfgnn.Adjacency.from_indices(
                        source = (df[0], source),
                        target = (df[1], target)))
            else:
                features = {}
                for col in df[2].columns[2:]:
                    if df[2][col].dtype == int:
                        df[2][col] = df[2][col].astype('int32')
                    elif df[2][col].dtype == float:
                        df[2][col] = df[2][col].astype('float32')
                    features[col] = df[2][col].values.reshape(len(df[2]),1)
                edge_sets[name] = tfgnn.EdgeSet.from_fields(
                    sizes = [len(df[2])],
                    features = features,
                    adjacency = tfgnn.Adjacency.from_indices(
                        source = (df[0], source),
                        target = (df[1], target)))

        if graph_dict['context'] != None:
            if type(graph_dict['context']) == 'float64':
                value = np.array([[value]],dtype='float32')
            elif type(graph_dict['context']) == 'int64':
                value = np.array([[value]],dtype='int32')
            else:
                value = graph_dict['context']

            context = tfgnn.Context.from_fields(
                sizes = [1],
                features ={
                    "context": value})

            graph_tensor = tfgnn.GraphTensor.from_pieces(
                context = context,
                node_sets = node_sets,
                edge_sets = edge_sets)
        else:
            graph_tensor = tfgnn.GraphTensor.from_pieces(
                node_sets = node_sets,
                edge_sets = edge_sets)
        return graph_tensor
    
    def from_networkx(self,nx_graph):
        graph_dict = {}
        if len(nx_graph.graph) > 0:
            graph_dict['context'] = [[nx_graph.graph['context']]]
        else:
            graph_dict['context'] = None

        node_dfs = {}
        node_dict = dict(nx_graph.nodes(data=True))
        node_types = set([value['node_type'] for key,value in node_dict.items()])
        for node in node_types:
            filter_dict = {k:v for k,v in node_dict.items() if node in v['node_type']}
            filter_df = pd.DataFrame.from_dict(filter_dict, orient='index')
            filter_df.index.name = node
            node_dfs[node] = filter_df.drop(columns=['node_type'])
        graph_dict['node_dfs'] = node_dfs

        edge_dfs = {}
        edge_df = nx.to_pandas_edgelist(nx_graph)
        for edge in edge_df['edge_type'].unique():
            filter_df = edge_df.loc[edge_df['edge_type']==edge].copy()
            edge_dfs[edge] = [filter_df.loc[0,'edge_source'],
                              filter_df.loc[0,'edge_target'],
                              filter_df.drop(columns=['edge_type','edge_source','edge_target'])]
        graph_dict['edge_dfs'] = edge_dfs
        
        graph_tensor = self.from_pandas(graph_dict)
        return graph_tensor
        
    def create_dataset(self,batch_size=32):
        if self.target[0] == 'node':
            features = self.graph_tensor.node_sets[self.target[1]].get_features_dict()
            label = features.pop(self.target[2])
            new_graph = self.graph_tensor.replace_features(node_sets={self.target[1]:features})
        elif self.target[0] == 'edge':
            features = self.graph_tensor.edge_sets[self.target[1]].get_features_dict()
            label = features.pop(self.target[2])
            new_graph = self.graph_tensor.replace_features(edge_sets={self.target[1]:features})
        else:
            features = self.graph_tensor.context.get_features_dict()
            label = features.pop(self.target[1])
            new_graph = self.graph_tensor.replace_features(context=features)
        dataset = tf.data.Dataset.from_tensors((new_graph,label))
        return dataset.batch(batch_size)
    
    def set_initial_node_state(self, node_set, node_set_name):
        feature_cnt = len(node_set.features)
        if feature_cnt == 0:
            return tfgnn.keras.layers.MakeEmptyFeature()(node_set)
        elif feature_cnt == 1:
            feature_name = list(node_set.features)[0]
            return tf.keras.layers.Dense(16)(node_set[feature_name])
        else:
            feature_layers = [tf.keras.layers.Dense(16)(layer) for _, layer in node_set.features.items()]
            return tf.keras.layers.Concatenate()(feature_layers)

    def set_initial_edge_state(self, edge_set, edge_set_name):
        feature_cnt = len(edge_set.features)
        if feature_cnt == 0:
            if self.target[0] == 'edge':
                return tfgnn.keras.layers.MakeEmptyFeature()(edge_set)
        elif feature_cnt == 1:
            feature_name = list(edge_set.features)[0]
            return tf.keras.layers.Dense(16)(edge_set[feature_name])
        elif feature_cnt > 1:
            feature_layers = [tf.keras.layers.Dense(16)(layer) for _, layer in edge_set.features.items()]
            return tf.keras.layers.Concatenate()(feature_layers)
        
    def dense_layer(self,units=64,l2_regularization=0.01,dropout_rate=0.1,activation='relu'):
        regularizer = tf.keras.regularizers.l2(l2_regularization)
        return tf.keras.Sequential([
            tf.keras.layers.Dense(units,
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer),
            tf.keras.layers.Dropout(dropout_rate)])
    
    def build_model(self,graph_spec,node_dict={"node":{"edge_dict":"source"}},graph_updates=3,
                    next_state_dim=64,message_dim=64,l2_reg=0.01,dropout=0.1,logit_size=1,loss='mse'):
        input_graph = tf.keras.layers.Input(type_spec=graph_spec)
        graph = input_graph.merge_batch_to_components()
        graph = tfgnn.keras.layers.MapFeatures(
            node_sets_fn=self.set_initial_node_state,
            edge_sets_fn=self.set_initial_edge_state)(graph)
        
        for i in range(graph_updates):
            node_sets = {node: tfgnn.keras.layers.NodeSetUpdate(
                {edge: tfgnn.keras.layers.SimpleConv(
                    sender_edge_feature=source,
                    message_fn = self.dense_layer(message_dim,l2_reg,dropout),
                    reduce_type="sum",
                    receiver_tag=tfgnn.TARGET) for edge, source in edge_dict.items()},
                tfgnn.keras.layers.NextStateFromConcat(
                    self.dense_layer(next_state_dim,l2_reg,dropout))) for node, edge_dict in node_dict.items()}
            
            if self.target[0] == 'edge':
                print(self.target[1])
                edge_sets = {self.target[1]: tfgnn.keras.layers.EdgeSetUpdate(
                    next_state = tfgnn.keras.layers.NextStateFromConcat(self.dense_layer(next_state_dim,l2_reg,dropout)))}
                graph = tfgnn.keras.layers.GraphUpdate(edge_sets = edge_sets,
                                                       node_sets = node_sets)(graph)
            else:
                graph = tfgnn.keras.layers.GraphUpdate(
                    node_sets = node_sets)(graph)

        if self.target[0] == 'context':
            print("context prediction")
            logit_stack = tfgnn.keras.layers.Pool(tfgnn.CONTEXT, "mean", node_set_name=list(node_dict)[0])(graph)
        elif self.target[0] == 'node':
            print("node prediction")
            #logit_stack = tfgnn.keras.layers.ReadoutFirstNode(node_set_name=list(node_dict)[0])(graph)
            logit_stack = graph.node_sets[list(node_dict)[0]][tfgnn.HIDDEN_STATE]
        elif self.target[0] == 'edge':
            print("edge prediction")
            logit_stack = graph.edge_sets[self.target[1]][tfgnn.HIDDEN_STATE]
        else:
            print("invalid target")
            
        if loss == 'mse':
            print("linear output")
            logits = tf.keras.layers.Dense(logit_size,activation='linear')(logit_stack)
        elif loss == 'binary_crossentropy':
            print('sigmoid output')
            logits = tf.keras.layers.Dense(logit_size,activation='sigmoid')(logit_stack)
        elif loss == 'categorical_crossentropy':
            print('softmax output')
            logits = tf.keras.layers.Dense(logit_size,activation='softmax')(logit_stack)
        else:
            print("invalid loss function")

        return tf.keras.Model(inputs=[input_graph], outputs=[logits])
    
    def train_model(self,params,trial=True):
        model = self.build_model(params['trainset'].element_spec[0],
                                 node_dict = params['node_dict'],
                                 graph_updates = params['graph_updates'],
                                 next_state_dim = params['next_state_dim'],
                                 message_dim = params['message_dim'],
                                 l2_reg = params['l2_reg'],
                                 dropout = params['dropout'],
                                 logit_size = params['logit_size'],
                                 loss = params['loss'])
        
        model.compile(tf.keras.optimizers.Adam(),
                      loss=params['loss'],
                      metrics=['Accuracy'])
        
        if params['valset'] == None:
            validation_data = None
            callbacks = None
        else:
            validation_data = params['valset']
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          mode='min',
                                                          verbose=1,
                                                          patience=params['patience'],
                                                          restore_best_weights=True)]
        model.fit(params['trainset'].repeat(),
                  validation_data=validation_data,
                  steps_per_epoch=params['steps_per_epoch'],
                  epochs=params['epochs'],
                  verbose=0,
                  callbacks = callbacks)
            
        if params['testset'] != None:
            loss = model.evaluate(params['testset'])[0]
        elif params['valset'] != None:
            loss = model.evaluate(params['valset'])[0]
        else:
            loss = model.evaluate(params['trainset'])[0]  
            
        if trial == True:
            sys.stdout.flush()
            print('trial complete')
            return {'loss': loss, 'status': STATUS_OK}
        else:
            print('loss:',loss)
            return model
        
    def GCNN_trials(self,params,max_evals):
        hyperparams = {
            'graph_updates': hp.choice('graph_updates',[2,3,4]),
            'next_state_dim': hp.choice('next_state_dim',[16,32,64,128]),
            'message_dim': hp.choice('message_dim',[16,32,64,128]),
            'l2_reg': hp.uniform('l2_reg',0.0,0.3),
            'dropout': hp.choice('dropout',[0,0.125,0.25,0.375,0.5]),
            'logit_size': 1,
            'node_dict': None,
            'loss': 'mse',
            'steps_per_epoch':10,
            'epochs': 100,
            'dataset': self.create_dataset(batch_size=32),
            'valset': None,
            'patience':3
        }
            
        for param in hyperparams.keys():
            if param not in params.keys():
                params[param] = hyperparams[param]
        
        best = fmin(self.train_model, params, algo=tpe.suggest, max_evals=max_evals, trials=Trials())
        print('best: ', best)
