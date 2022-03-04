using System;
using System.Net.Http;
using System.Threading.Tasks;
 
class Program
{
    static async Task Main(string[] args)
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
 
        // Pass the text to the Web service
        var client = new HttpClient();
        var url = $"http://localhost:5000/analyze?text={text}";
        var response = await client.GetAsync(url);
        var score = await response.Content.ReadAsStringAsync();
 
        // Show the sentiment score
        Console.WriteLine(score);
    }
}