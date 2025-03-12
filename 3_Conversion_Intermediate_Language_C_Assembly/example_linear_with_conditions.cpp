class Linear{

    public:

        //
        int in_features;
        int out_features;
        bool bias;

        //
        Tensor A;
        Tensor b;

        //
        Linear(int in_features, int out_features, bool bias):
            in_features(in_features), out_features(out_features), bias(bias) {}

        //
        Tensor forward(Tensor X){

            Tensor X1 = matmult(X, A);

            Tensor Y;
            if(bias){
                Y = with_bias(X1);
            }
            else{
                Y = without_bias(X1);
            }

            return Y;

        }

        //
        Tensor with_bias(Tensor X){
            return matadd(X, b)
        }

        //
        Tensor without_bias(Tensor X){
            return X;
        }
}