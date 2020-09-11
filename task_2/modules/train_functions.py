import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


def check_accuracy(model, X, y_true):

    ''' Compute accuracy for model and X with given y_true '''

    out = model(X)
    out = (out.flatten() > 0.5).float()
    return (y_true == out).float().mean().item()


def plot_accs_and_loss(results, title):
    '''
    Based on the output of the train_nn function, create 2 plots: first shows a loss history, 
    second shows a history of accuracy on training data and a history of accuracy on validation data  
    '''

    fig = plt.figure()

    ax = fig.add_subplot(2, 1, 1)
    ax.plot(results[1], 'o-', label='train acc')
    ax.plot(results[2], 'o-', label='val acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.title(title)
    plt.legend()

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(results[0], 'o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
 
    plt.show()


def train_nn(train_parameters, X_train, y_train, X_val, y_val, 
             num_epochs=10, batch_size=50, check_every=None,
             save_model=True, checkpoint_filename='model.pth', verbose=True):
    '''
    Train Neural Network.
    
    Inputs:
    - train_parameters: a dictionary, that contains following keys:
        * 'model': object of a class inheriting nn.Module
        * 'criterion': loss function
        * 'optimizer': object that implement a gradient descent (from torch.optim)
    - X_train, y_train: train data
    - X_val, y_val: validation data
    - num_epochs: number of epochs for NN training 
    - batch_size: size of batch that function uses to perform training step
    - check_every: every check_every iteration function checks loss, 
                   accuracy on training and validation data. If None, then checks once per epoch.
    - save_model: if True, the function save a model with the highest accuracy on validation data
    - checkpoint_filename: where to save a model with the highest accuracy on validation data
    - verbose: if True, print an additional information
    
    Outputs:
    - loss_history: a list with losses, computing every check_every iteration
    - train_acc_history: a list with accuracy on training data
    - val_acc_history: a list with accuracy on validation data
    - best_val: the highest accuracy on validation data
    '''
  
    model = train_parameters['model']
    criterion = train_parameters['criterion']
    optimizer = train_parameters['optimizer']

    train_size = X_train.shape[0]
    if check_every is None:
        check_every = train_size // batch_size
    loss_history, train_acc_history, val_acc_history = [], [], []
    best_val = 0

    if verbose:
        print('Start training')

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i in range(train_size // batch_size):
            
            # generate tensor of random indices with size batch_size
            batch_indices = torch.randint(0, train_size, (batch_size, ))
            
            # create mini-batch of training data
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices].view(-1, 1)
            
            # clear gradients of optimizer
            optimizer.zero_grad()

            # forward pass 
            out = model(X_batch)
            
            # compute loss
            loss = criterion(out.float(), y_batch.float())
            
            # backward pass
            loss.backward()
            
            # update model's weights 
            optimizer.step()

            # accumulate loss history for current epoch
            running_loss += loss.item()
            if (i + 1) % check_every == 0:
                loss_history.append(running_loss / (i + 1))

                train_acc = check_accuracy(model, X_train, y_train)
                train_acc_history.append(train_acc)
                
                val_acc = check_accuracy(model, X_val, y_val)
                val_acc_history.append(val_acc)
                if val_acc > best_val:
                    best_val = val_acc
                    if save_model:
                        torch.save(model.state_dict(), checkpoint_filename)
                if verbose:
                    print(f'Epoch [{epoch+1}/{num_epochs}] it {i+1}: loss {running_loss / (i+1)} - ' +\
                          f'train acc {train_acc} - val acc {val_acc}')
    if verbose:
        print('Finishing training')
        print('Best accuracy on validation set: ', best_val, '\n')
    return ([loss_history, train_acc_history, val_acc_history], best_val)


def cross_validation_score(model_class, X, y, 
             num_epochs=10, batch_size=200, check_every=None,
             num_folds=5, shuffle=True):
    '''
    Performs cross-validation.
    
    Inputs:
    - model_class: class of model that subclass nn.Module
    - X, y: tensors with input data
    - num_epochs: number of training epochs
    - batch_size: size of batch that function uses to perform training step
    - check_every: every check_every iteration function checks loss, 
                   accuracy on training and validation data. If None, then checks once per epoch.
    - num_folds: number of cross-validation folds
    - shuffle: if True, data will be shuffled before splitting
    
    Outputs:
    - val_results: a list with the highest validation accuracy for each fold
    '''
    
    # how many objects every fold contains
    num_in_fold = X.shape[0] // num_folds
    val_results = []
    
    if shuffle:
        # shuffle data before split 
        shuffled_indices = torch.randperm(X.shape[0])
        X = X[shuffled_indices]
        y = y[shuffled_indices]
    
    for i in range(num_folds):

        print(f'Fold {i+1} ->')

        # splitting process
        X_val = X[i * num_in_fold: (i + 1) * num_in_fold]
        y_val = y[i * num_in_fold: (i + 1) * num_in_fold]
        if i == 0:
            X_train = X[(i + 1) * num_in_fold:]
            y_train = y[(i + 1) * num_in_fold:]
        elif i == num_folds - 1:
            X_train = X[:i * num_in_fold]
            y_train = y[:i * num_in_fold]
        else:
            X_train = torch.cat((X[:i * num_in_fold], X[(i + 1) * num_in_fold:]))
            y_train = torch.cat((y[:i * num_in_fold], y[(i + 1) * num_in_fold:]))
        
        # create NN model
        net = model_class()
        if torch.cuda.is_available():
            net = net.cuda()

        criterion = nn.BCELoss()
        optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999))

        train_parameters = {
            'model': net,
            'criterion': criterion,
            'optimizer': optimizer
        }
        
        results, best_val = train_nn(train_parameters, X_train, y_train, X_val, y_val, 
                                     num_epochs=num_epochs, batch_size=batch_size,
                                     check_every=check_every, save_model=False, verbose=True)
        val_results.append(best_val)
        plot_accs_and_loss(results, f'Fold {i+1}')
    
    return val_results