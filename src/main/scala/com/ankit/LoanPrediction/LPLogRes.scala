package com.ankit.LoanPrediction

import scala.collection.mutable.ListBuffer

import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.classification.LinearSVCModel
import org.apache.spark.ml.classification.NaiveBayes



object LPLogRes {
  
  var regParams = Array(0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12,10.24)
  var maxIterations = Array(5,10,100,1000)
  
  var loss : ListBuffer[Double] = new ListBuffer()
  var cvLoss : ListBuffer[Double] = new ListBuffer()
  def bestModel(trainSet : DataFrame) : CrossValidatorModel={
    
   
    
    val lr = new LogisticRegression().setFeaturesCol("features")
                         .setLabelCol("label")
                         .setPredictionCol("predictedLabel")
                         .setThreshold(0.4)
    val evaluator = new MulticlassClassificationEvaluator()
                  .setLabelCol("label")
                  .setPredictionCol("predictedLabel")
                  .setMetricName("accuracy")
    val paramMaps = new ParamGridBuilder()
                         // .addGrid(lr.regParam, regParams)
                          //.addGrid(lr.maxIter, maxIterations)
                          .build()
    val cv = new CrossValidator().setEstimator(lr)
                  .setEvaluator(evaluator)
                  .setEstimatorParamMaps(paramMaps)
  
    val model = cv.fit(trainSet)
    return model;
    
  }
  
   def main(args : Array[String]) {
   val spark = SparkSession.builder().getOrCreate();
   import spark.implicits._
   val trainFile = spark.read.format("csv").option("header", "true").load(args(0));
   val testFile = spark.read.format("csv").option("header", "true").load(args(1))
   val dataPrep = new DataPreprator()
   var trainData = dataPrep.prepareFeatures(trainFile)
   trainData = dataPrep.prepareLabel(trainData)
   trainData.persist();
   val testData = dataPrep.prepareFeatures(testFile)
   
   val model =  bestModel(trainData);
   val rslt = model.transform(testData);
   val Array(trnData, cvData)= trainData.randomSplit(Array(0.7,0.3));
   val evaluator = new MulticlassClassificationEvaluator()
                  .setLabelCol("label")
                  .setPredictionCol("predictedLabel")
                  .setMetricName("accuracy")
    val trnAccuracy = evaluator.evaluate(model.transform(trnData))
    val cvAccuracy = evaluator.evaluate(model.transform(cvData))
    
                  
   rslt.select("Loan_ID","predictedLabel").show()
   rslt.groupBy("predictedLabel").count.show()
   
     val testSetOutput = model.transform(trainData).toDF();
     val testOutput = testSetOutput;
     val labelReverse = new IndexToString().setInputCol("predictedLabel")
                                       .setOutputCol("Predicted Output")
                                       .setLabels(dataPrep.labelIndexer)
     labelReverse.transform(rslt).select($"Loan_ID", $"Predicted Output".as("Loan_Status")).write.option("header",true).csv(args(2))                                
                                      
     val reversedOutput = labelReverse.transform(testOutput)
     reversedOutput.show()
     reversedOutput.groupBy("Predicted Output").count.show()
     reversedOutput.groupBy("Loan_Status","Predicted Output", "Credit_History").count.show()
     println("Training Data Accuracy : "+trnAccuracy*100)
     println("CV Data Accuracy : "+cvAccuracy*100)
    
   
   
  }

}
