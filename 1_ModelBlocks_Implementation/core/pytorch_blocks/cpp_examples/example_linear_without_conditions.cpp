class Linear{

    public:

        //
        int in_features;
        int out_features;
        int bias;

        //
        Tensor A;
        Tensor b;

        //
        Linear(int in_features, int out_features, int bias):
            in_features(in_features), out_features(out_features), bias(bias) {}

        //
        Tensor forward(Tensor X){

            Tensor X1 = matmult(X, A);

            Tensor B = matpond(b, bias);

            Tensor Y = matadd(X, B);

            return Y;

        }
}