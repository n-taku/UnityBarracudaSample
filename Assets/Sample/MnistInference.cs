using Unity.Barracuda;

public class MnistInference
{
    readonly IWorker worker;
    
    public MnistInference(NNModel modelAsset)
    {
        var runtimeModel = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.CSharpBurst, runtimeModel);
    }

    //推論
    //Mnistは28x28のfloat値(0~1)のinputで推論できる，左上が原点で右下に向かう座標系
    public int Inference(float[] inputFloats)
    {
        //推論する
        var scores = InferenceOnnx(inputFloats);

        //最大のIndexを求める．Indexが推論した数字
        var maxScore = float.MinValue;
        int maxIndex = 0;
        for (int i = 0; i < scores.Length; i++)
        {
            float score = scores[i];
            if (maxScore < score)
            {
                maxScore = score;
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    private float[] InferenceOnnx(float[] input)
    {
        var inputTensor = new Tensor(1, 28, 28, 1, input);
        worker.Execute(inputTensor);
        var outputTensor = worker.PeekOutput();
        var outputArray = outputTensor.ToReadOnlyArray();
        
        inputTensor.Dispose();
        outputTensor.Dispose();

        return outputArray;
    }

    ~MnistInference()
    {
        worker.Dispose();
    }
}
