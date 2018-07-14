package com.ankit.LoanPrediction

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql
import org.apache.spark.ml.param.StringArrayParam
import org.apache.spark.ml.feature.Interaction

class DataPreprator {
  
  var labelIndexer:Array[String] =Array("")
  
  def vectorFormation(df: DataFrame, colName : String) : DataFrame ={
    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._;
    val indexer = new StringIndexer()
                .setInputCol(colName)
                .setOutputCol(colName+"_Index")
                      .fit(df)
    val indexed = indexer.transform(df)
    //indexed.show()
    val encoder = new OneHotEncoder()
              .setInputCol(colName+"_Index")
               .setOutputCol(colName+"_Vector")                   
    val encoded = encoder.transform(indexed)
    //encoded.show()
    return encoded;
  }
  
  def fillMissingRecords(colName : String, value : String, df :DataFrame):DataFrame = {
    val updatedDF = df.na.fill(value, Array(colName))
    
    return updatedDF;
  }
  
  def featureScaler(df : DataFrame, featureName : String) : DataFrame ={
    val scaler = new StandardScaler()
                      .setInputCol(featureName)
                      .setOutputCol("scaled"+featureName)
                      .setWithStd(false)
                      .setWithMean(true)
    val scalerModel = scaler.fit(df)    
    val scaledData = scalerModel.transform(df)
    return scaledData;
  }
  
  def prepareFeatures(df: DataFrame) : DataFrame ={
    val avgAIncm = df.agg(avg("ApplicantIncome")).head.getDouble(0)
    val avgCIncm = df.agg(avg("CoapplicantIncome")).head.getDouble(0)
    val avgLoanAmount = df.agg(avg("LoanAmount")).head.getDouble(0)
    
    var updtdDF = df.na.fill("Male",Array("Gender"))
    updtdDF = updtdDF.na.fill("Yes",Array("Married"))
    updtdDF = updtdDF.na.fill("0".toString,Array("Dependents"))
    updtdDF = updtdDF.na.fill("No",Array("Self_Employed"))
    updtdDF = updtdDF.na.fill("360".toString,Array("Loan_Amount_Term"))
    
    
    updtdDF = updtdDF.withColumn("ApplicantIncome",col("ApplicantIncome").cast(sql.types.DoubleType)).na.fill(avgAIncm)
    updtdDF = updtdDF.withColumn("CoapplicantIncome",col("CoapplicantIncome").cast(sql.types.DoubleType) ).na.fill(0)
    updtdDF = updtdDF.withColumn("LoanAmount",log(col("LoanAmount").cast(sql.types.DoubleType) )).na.fill(avgLoanAmount)
    updtdDF = updtdDF.withColumn("TotalIncome",log(col("ApplicantIncome")+col("CoapplicantIncome"))).na.fill(0)
    //updtdDF = updtdDF.withColumn("Incm_LoanAmount",log(col("LoanAmount").cast(sql.types.DoubleType)*col("TotalIncome")))
    updtdDF = updtdDF.withColumn("EMI",log(col("LoanAmount").cast(sql.types.DoubleType)/col("Loan_Amount_Term").cast(sql.types.DoubleType))*1000)
    
    updtdDF = updtdDF.withColumn("Credit_History",col("Credit_History").cast(sql.types.DoubleType) ).na.fill(1.0)
    
    
    updtdDF = updtdDF.withColumn("income_ch",col("TotalIncome")*col("Credit_History")).na.fill(0)
    
    updtdDF = vectorFormation(updtdDF,"Gender")
    updtdDF = vectorFormation(updtdDF,"Married")
    updtdDF = vectorFormation(updtdDF,"Dependents")
    updtdDF = vectorFormation(updtdDF,"Education")
    updtdDF = vectorFormation(updtdDF,"Self_Employed")
    updtdDF = vectorFormation(updtdDF,"Property_Area")
    
    
    val featureCols = Array("income_ch","EMI","Education_Index","Married_Index","Property_Area_Index","Dependents_Index")
    val integrator = new VectorAssembler()
                          .setInputCols(featureCols)
                          .setOutputCol("features")
      
    updtdDF = integrator.transform(updtdDF)
    return updtdDF
  
  }
  
  def prepareLabel(df :DataFrame) : DataFrame = {
    
    

    val indexer = new StringIndexer()
        .setInputCol("Loan_Status")
        .setOutputCol("label")
    
    val indexedModel = indexer.fit(df)
    labelIndexer = indexedModel.labels
    val indexed = indexedModel.transform(df)
    
   
    
    return indexed
  }
  
}