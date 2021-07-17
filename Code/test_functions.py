def HeatMap_ConfMatrix(BestModel,x_test,y_test,y_labels):

  prediction = np.argmax(BestModel.predict(x_test), axis=-1)
  prediction = prediction.reshape(prediction.shape[0],1)
  prediction = prediction+1

  y_true = np.argmax(y_test, axis=-1)
  y_true = y_true+1
  clf_report = classification_report(y_true, prediction)
  print(clf_report)

  labels = unique(y_true)
  target_names = unique(y_labels)

  clf_report = classification_report(y_true,
                                    prediction,
                                    labels,
                                    target_names,
                                    output_dict=True)

  sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, cmap="Blues")