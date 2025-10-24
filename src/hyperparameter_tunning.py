from sklearn.model_selection import GridSearchCV



def hypertune(model,param_grid,x_train,y_train,cv=5,scoring='accuracy',verbose=True):

    grid = GridSearchCV(model, param_grid=param_grid, cv=cv, scoring=scoring)
    grid.fit(x_train, y_train)
    
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_score = grid.best_score_
    
    if verbose:
        print(f"Model: {model.__class__.__name__}")
        print("Best Parameters:", best_params)
        print(f"Best CV Score: {best_score:.4f}")
        print("-"*40)
    
    return best_model, best_params, best_score