package Final.NareshB

import org.apache.spark.SparkContext._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import edu.stanford.nlp.pipeline._
import edu.stanford.nlp.util._
import edu.stanford.nlp.ling.CoreAnnotations._
import scala.collection.JavaConversions._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.SQLContext
import org.apache.hadoop.conf.Configuration
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.IDF
import org.apache.spark.mllib.linalg.{Vectors, Vector => MLLibVector}
import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Matrix, SingularValueDecomposition}
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.{Word2Vec, Word2VecBase, Word2VecModel, VectorAssembler, OneHotEncoder}
import org.apache.spark.ml.classification.BinaryLogisticRegressionSummary
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import com.databricks.spark.csv._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.Row

object Regkill {
    val sc = new SparkContext(new SparkConf().setAppName("Regression Attack Severity").setMaster("local[*]").set("spark.driver.memory","2g"))
  val sqlContext = new SQLContext(sc)
  import sqlContext.implicits._
  def main(args: Array[String]) = {
    val data = sqlContext.read.textFile("hdfs://quickstart.cloudera:8020/user/cloudera/final/gtd_final").map({
        text => {
          val tuple = mkAttackTuple(text)
          val pipe = startPipe()
          new TerrorCaseLemma(tuple.iyear, tuple.imonth, tuple.country, tuple.success, tuple.multiple, tuple.suicide ,tuple.attackType, getLemmas(tuple.source1_txt,pipe) ,tuple.nkill)
        }
      })
      val terrorDF = data.toDF("iyear","imonth","country","success","multiple","suicide","attackType","source1","nkill")
      terrorDF.cache()
      val filteredDF = terrorDF.filter(not($"iyear" === "iyear"))
        .filter(not($"iyear" === "BR"))
        .select("iyear","imonth","country","success","multiple","suicide","attackType","source1","nkill")
      filteredDF.show()
      
      val full = filteredDF.toDF("iyear","imonth","country","success","multiple","suicide","attackType","source1","nkill")
     
      
      val siMonth = new StringIndexer().setInputCol("imonth").setOutputCol("lblMonth")
      val siCountry = new StringIndexer().setInputCol("country").setOutputCol("lblCountry")
      val siSuccess = new StringIndexer().setInputCol("success").setOutputCol("lblSuccess")
      val siMultiple = new StringIndexer().setInputCol("multiple").setOutputCol("lblMultiple")
      val siSuicide = new StringIndexer().setInputCol("suicide").setOutputCol("lblSuicide")
      val siAT = new StringIndexer().setInputCol("attackType").setOutputCol("lblAT")
      val siNK = new StringIndexer().setInputCol("nkill").setOutputCol("label")
      val ixMonth = new OneHotEncoder()
        .setInputCol("lblMonth")
        .setOutputCol("ixMonth")
      val ixCountry = new OneHotEncoder()
        .setInputCol("lblCountry")
        .setOutputCol("ixCountry")
     val ixSuccess = new OneHotEncoder()
        .setInputCol("lblSuccess")
        .setOutputCol("ixSuccess")
      val ixMultiple = new OneHotEncoder()
        .setInputCol("lblMultiple")
        .setOutputCol("ixMultiple")
      val ixSuicide = new OneHotEncoder()
        .setInputCol("lblSuicide")
        .setOutputCol("ixSuicide")
      val ixAT = new OneHotEncoder()
        .setInputCol("lblAT")  
        .setOutputCol("ixAT")
      
      val termLimit = 10000
      val countVectorizer = new CountVectorizer().setInputCol("source1").setOutputCol("sc1Freqs").setVocabSize(termLimit)
 //       countVectorizer.show()
      
      
      val idf = new IDF().setInputCol("sc1Freqs").setOutputCol("idfFeatures")//configuring the IDF to recieve an input column of term frequencies and output a column of inverse document frequencies
//   idf.show()
        
      val w2v = new Word2Vec().setInputCol("source1").setOutputCol("w2vFeatures")
  //      w2v.show()
    
      val va = new VectorAssembler().setInputCols(Array("ixMonth","ixCountry","ixSuccess","ixMultiple","ixSuicide","ixAT","w2vFeatures","idfFeatures")).setOutputCol("features")
//      va.show()
       
      val pipeline = new Pipeline().setStages(Array(siMonth, siCountry, siSuccess,siMultiple, siSuicide, siAT, siNK, ixMonth, ixCountry, ixSuccess,ixMultiple, ixSuicide, countVectorizer,ixAT, idf, w2v, va))
      val fmtDF = pipeline.fit(filteredDF).transform(filteredDF)
      val seed = 20564
      val Array(training, test) = fmtDF.randomSplit(Array(0.7, 0.3), seed)
      val lr = new LogisticRegression().setFeaturesCol("features").setLabelCol("label")
      val lrDF = lr.fit(training).transform(test)
      lrDF.select("iyear","imonth", "probability", "label", "prediction", "nkill")
      .collect()//converts the dataframe into an array of Row objects
      .foreach{
      case Row(iyear: String, imonth: String, probability: MLVector, label: Double, prediction: Double, nkill: String) => 
        println(s"($iyear,$imonth, $nkill, $label) --> prob=$probability, pred=$prediction")
          }
    
    val lp = lrDF.select( "label", "prediction")
    val counttotal = lrDF.count()
    val correct = lp.filter($"label" === $"prediction").count()
    val wrong = lp.filter(not($"label" === $"prediction")).count()
    val accuracy = correct.toDouble/counttotal.toDouble
    val truep = lp.filter($"prediction" === 1.0).filter($"label" === $"prediction").count() 
    val truen = lp.filter($"prediction" === 0.0).filter($"label" === $"prediction").count()
    val falseN = lp.filter($"prediction" === 0.0).filter(not($"label" === $"prediction")).count() 
    val falseP = lp.filter($"prediction" === 1.0).filter(not($"label" === $"prediction")).count() 
    val ratioWrong=wrong.toDouble/counttotal.toDouble
    val ratioCorrect=correct.toDouble/counttotal.toDouble
    val precision = truep.toDouble/counttotal.toDouble
    val recall = truep.toDouble/(truep.toDouble+falseN.toDouble)
    println(s"Correct: ${correct}")
    println(s"Wrong: ${wrong}")
    println(s"Accuracy: ${accuracy}")
    println(s"True Positive Count: ${truep}")
    println(s"True Negative Count: ${truen}")
    println(s"False Positive Count: ${falseP}")
    println(s"False Negative Count: ${falseN}")
    println(s"Precision Ratio: ${precision}")
    println(s"Recall Ratio: ${recall}")
        
    }
    
     def getLemmas(text: String, cNLP: StanfordCoreNLP): Seq[String] = {
     val document = new Annotation(text)
    cNLP.annotate(document)
    val lemmas = new scala.collection.mutable.ArrayBuffer[String]()
    
    val sentences = document.get(classOf[SentencesAnnotation])
    for (sentence <- sentences; 
    token <- sentence.get(classOf[TokensAnnotation])) {
      val lemma = token.get(classOf[LemmaAnnotation])
      if (lemma.length > 2 && isLetters(lemma)) {
        lemmas += lemma.toLowerCase 
      }
    }
    lemmas
  }
     
      def startPipe(): StanfordCoreNLP = {
    val pipe = new StanfordCoreNLP(PropertiesUtils.asProperties(
        "annotators", "tokenize,ssplit,pos,lemma"))
    pipe
  }
     
   def isLetters (str: String): Boolean = {
    str.forall(x => Character.isLetter(x))
  }
      //initial preprocessor to extract fields from the comma separated data
  def mkAttackTuple(line: String) :TerrorCase = {
    val terrorFields = line.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)")
    
    
    if (terrorFields.size == 9 )
    {
      val iyear = terrorFields.apply(0)
      val imonth = terrorFields.apply(1)
      val country = terrorFields.apply(2)
      val success = terrorFields.apply(3)
      val multiple = terrorFields.apply(4)
      val suicide = terrorFields.apply(5)
      val attackType = terrorFields.apply(6)
      val source1 = terrorFields.apply(7)
      val nkill = terrorFields.apply(8)
      TerrorCase(iyear, imonth, country, success,multiple, suicide, attackType, source1,nkill)
    }
    else
    {
      val iyear = "BR"
      val imonth = "BR"
      val country = "BR"
      val success = "BR"
      val multiple = "BR"
      val suicide = "BR"
      val attackType = "BR"
      val source1 = "BR"
      val nkill = "BR"
      TerrorCase(iyear, imonth, country, success, multiple, suicide, attackType, source1, nkill)
    }
    
  }
}