# Transform dataset into columns named by all aggregation levels
def agg_data(df):
    
    df['dept_id'] = df['store_id'] + '_' + df['dept_id']
    df['item_id'] = df['store_id'] + '_' + df['item_id']

    state_data = df.drop(['item_id','dept_id','store_id'],axis=1).groupby('state_id').sum()
    store_data = df.drop(['item_id','dept_id','state_id'],axis=1).groupby('store_id').sum()
    dept_data = df.drop(['item_id','store_id','state_id'],axis=1).groupby('dept_id').sum()

    agg_data = pd.concat([state_data,store_data,dept_data]).T
    agg_data['total'] = agg_data['CA'] + agg_data['TX'] + agg_data['WI']
    
    cols = [agg_data.columns[-1]] + [col for col in agg_data if col != agg_data.columns[-1]]
    agg_data = agg_data[cols]
    
    return agg_data

# define a hierarchy
states = data.state_id.unique()
stores = data.store_id.unique()
depts = data.dept_id.unique()

# build hierarchy tree as a dictionary
total = {'total': list(states)}
state_h = {k: [v for v in stores if v.startswith(k)] for k in states}
store_h = {k: [v for v in depts if v.startswith(k)] for k in stores}

hierarchy = { **total,**state_h, **store_h}

tree = HierarchyTree.from_nodes(nodes=hierarchy, df=train)

# extract summing matrix
sum_mat, sum_mat_labels = hts.functions.to_sum_mat(tree)


# store pre-computed base forecasts with the same format in a dictionary
pred_dict = collections.OrderedDict()
for label in sum_mat_labels:
    pred_dict[label] = pd.DataFrame(data=forecasts[label].values, columns=['yhat'])

# reconcile forecasts with OLS optimal reconciliation
OLS_model = hts.functions.optimal_combination(pred_dict, sum_mat, method='OLS', mse={})
OLS_revised = pd.DataFrame(data=OLS_model[0:,0:],index=forecasts.index,columns=sum_mat_labels)
OLS_pred = OLS_revised[forecasts.columns]
OLS_pred = OLS_pred.iloc[:, :34]