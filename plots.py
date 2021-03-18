def plot_gain_lift(model):
    model_pred = model.predict(X_test)
    model_proba = model.predict_proba(X_test)

    temp = pd.DataFrame({
      'actual': y_test, 
      'p(0)': [p[0] for p in model_proba],
      'p(1)': [p[1] for p in model_proba],
      'predicted': model_pred,
    })

    temp = temp.sort_values(by=['p(1)'], ascending=False)
    
    plt.style.use('fivethirtyeight')
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    
    gainsChart(temp.actual, ax=axes[0])
    liftChart(temp.actual, title=False, ax=axes[1])
    
    plt.suptitle(str(model).split('(')[0])
    plt.show()
    
    
def plot_roc_auc(real, pred, model):
    # compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(real, pred)
    roc_auc = auc(fpr, tpr)
    plt.style.use("fivethirtyeight")
    plt.figure(figsize=[8, 6])
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title(str(model).split('(')[0])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend(loc="lower right")
    plt.show()