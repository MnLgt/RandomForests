from fastai.tabular.all import *


# def update_patch(self, obj):
#     clear_output(wait=True)
#     self.display(obj)

# DisplayHandle.update = update_patch

# import warnings
# warnings.filterwarnings('ignore',category=FutureWarning)


# update pandas settings to supress scientific notation

# pd.set_option('display.float_format', lambda x: '%.3f' % x)


def split_df(df, train_pct=0.8):
    # Splitting the data into training and test sets
    total = len(df)
    test_split = int(total * 0.1)

    # Test DF - Use the latest 10% of the data for testing
    test_df = df.tail(test_split).copy()

    # Train DF
    train_df = df[~df.index.isin(test_df.index)].copy()
    train_df.reset_index(drop=True, inplace=True)
    return train_df, test_df


def train_model(train_df, cont_names, y_name, model_path, epochs, train=True, layers=[200, 100]):
    # Get the Train / Validation splits
    splits = RandomSplitter(valid_pct=0.2)(range_of(train_df))

    if train:
        # Creating a TabularPandas object
        procs = [Categorify, FillMissing, Normalize]
        to = TabularPandas(
            train_df, procs, cat_names=[], cont_names=cont_names, y_names=y_name, splits=splits
        )

        # Creating a dataloader
        dls = to.dataloaders(bs=64)

        # Creating a learner and training the model
        learn = tabular_learner(dls, layers=layers, metrics=rmse)

        learn.fit_one_cycle(epochs)
        learn.recorder.plot_loss()
        learn.export(model_path)
    else:
        learn = load_learner(model_path, cpu=False)

    return learn

# Create a function that performs the above steps


def get_preds(df, columns, learn, pred_column='preds'):
    # Get the predictions for the entire dataset
    df_loader = learn.dls.test_dl(df[columns])

    # Get the predictions
    predictions, _ = learn.get_preds(dl=df_loader, with_decoded=False)
    preds = predictions.cpu().numpy()
    df[pred_column] = preds
    return df
