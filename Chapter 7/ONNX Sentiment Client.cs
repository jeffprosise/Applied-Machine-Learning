using System;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Generic;
using System.Linq;

class Program
{
    static void Main(string[] args)
    {
        string text;
 
        // Get the text to analyze
        if (args.Length > 0)
        {
            text = args[0];
        }
        else
        {
            Console.Write("Text to analyze: ");
            text = Console.ReadLine();
        }


        // Create the model and pass the text to it
        var tensor = new DenseTensor<string>(new string[]
            { text }, new int[] { 1, 1 });

        var input = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor<string>("string_input", tensor)
        };

        var session = new InferenceSession("sentiment.onnx");
        var output = session.Run(input)
            .ToList().Last().AsEnumerable<NamedOnnxValue>();

        var score = output.First().AsDictionary<Int64, float>()[1];
        
        // Show the sentiment score
        Console.WriteLine(score);
    }
}
