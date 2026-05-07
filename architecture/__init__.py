def model_generator(method, opt=None):
    
    if method=='hyper':
        if opt.stage==3:
            from .HyperBME import HyperBME

           
        model = HyperBME(opt).cuda()  
    else:
        print(f'Method {method} is not defined !!!!')

    return model


