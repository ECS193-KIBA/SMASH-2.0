function ViewMediumTreeModel()
    model = load('MediumTreeModel.mat','MediumTree');
    view(model.MediumTree.ClassificationTree,"Mode","graph")
end