import javafx.scene.chart.ScatterChart
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}
import org.apache.spark.sql.functions.{avg, col, stddev_pop, stddev_samp, udf, _}
import org.apache.spark.ml.feature.{MinMaxScaler, VectorAssembler}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.stat.ChiSquareTest



object main {

    def main(args: Array[String]): Unit = {

        val spark: SparkSession = SparkSession.builder().config("spark.master", "local[*]").getOrCreate()
        val sc: SparkContext = spark.sparkContext
        val sqlContext: SQLContext = spark.sqlContext
//        LogManager.getRootLogger.setLevel(Level.ERROR)
        import sqlContext.implicits._
        val t1 = System.nanoTime


        val dataDF = spark.read.option("header", "false").csv("data\\data-example.txt")

        // -------------------------------------Preprocessing-----------------------------------------------------------
        val doubleDF = dataDF
          .withColumn("x", col("_c0").cast("Double"))
          .withColumn("y", col("_c1").cast("Double"))
          .select("x", "y")

        val filteredDF = doubleDF.filter(row => (row(0) != null && row(1) != null))

        val vectorizeCol = udf( (v:Double) => Vectors.dense(Array(v)))
        val semiVectorizedDF = filteredDF.withColumn("vec_x", vectorizeCol(filteredDF("x")))
        val vectorizedDF = semiVectorizedDF.withColumn("vec_y", vectorizeCol(semiVectorizedDF("y")))

        val scale = new MinMaxScaler()
          .setInputCol("vec_x")
          .setOutputCol("scaled_x")
          .setMax(1.0)
          .setMin(0.0)

        val semiScaledDF = scale.fit(vectorizedDF).transform(vectorizedDF)

        scale.setInputCol("vec_y")
          .setOutputCol("scaled_y")

        val scaledDF = scale.fit(semiScaledDF).transform(semiScaledDF)
          .select("x","y", "scaled_x", "scaled_y")
          .withColumnRenamed("scaled_x", "xv")
          .withColumnRenamed("scaled_y", "yv")

        // --------------------------------------------KMeans-----------------------------------------------------------
        val assembler = new VectorAssembler()
          .setInputCols(Array("xv", "yv"))
          .setOutputCol("features")

        val output = assembler.transform(scaledDF)

        val model = new KMeans().setK(5).setSeed(3L).setMaxIter(1000).fit(output)
        val predictions = model.transform(output).select("x", "y", "prediction", "features")

//        predictions.select("x", "y", "prediction").write.mode("overwrite").csv("predictions.csv")

        // --------------------------------------------Test Area 51-----------------------------------------------------
        val distFromCenter = udf((features: Vector, c: Int) => Vectors.sqdist(features, model.clusterCenters(c)))

        val distanceDF = predictions.withColumn("distance", distFromCenter(predictions("features"), predictions("prediction"))).orderBy($"distance")

        val cluster0DF = distanceDF.filter($"prediction" === 0).orderBy($"distance".desc)
        val cluster1DF = distanceDF.filter(row => row(2) == 1).orderBy($"distance".desc)
        val cluster2DF = distanceDF.filter(row => row(2) == 2).orderBy($"distance".desc)
        val cluster3DF = distanceDF.filter(row => row(2) == 3).orderBy($"distance".desc)
        val cluster4DF = distanceDF.filter(row => row(2) == 4).orderBy($"distance".desc)

//        cluster0DF.agg(stddev_pop("distance")).show()
        println("Outliers detected based on neighbors density method: ")
        val aRandomValue = MyUtility.neighborDensity(spark, cluster0DF)
        aRandomValue.foreach(row => println(row))
        val aRandomValue1 = MyUtility.neighborDensity(spark, cluster1DF)
        aRandomValue1.foreach(row => println(row))
        val aRandomValue2 = MyUtility.neighborDensity(spark, cluster2DF)
        aRandomValue2.foreach(row => println(row))
        val aRandomValue3 = MyUtility.neighborDensity(spark, cluster3DF)
        aRandomValue3.foreach(row => println(row))
        val aRandomValue4 = MyUtility.neighborDensity(spark, cluster4DF)
        aRandomValue4.foreach(row => println(row))

//        println("Outliers detected based on neighbors density method with Normal Distribution: ")
//        val aRandomValue = MyUtility.neighborDensityND(spark, cluster0DF)
//        aRandomValue.foreach(row => println(row))
//        val aRandomValue1 = MyUtility.neighborDensityND(spark, cluster1DF)
//        aRandomValue1.foreach(row => println(row))
//        val aRandomValue2 = MyUtility.neighborDensityND(spark, cluster2DF)
//        aRandomValue2.foreach(row => println(row))
//        val aRandomValue3 = MyUtility.neighborDensityND(spark, cluster3DF)
//        aRandomValue3.foreach(row => println(row))
//        val aRandomValue4 = MyUtility.neighborDensityND(spark, cluster4DF)
//        aRandomValue4.foreach(row => println(row))


//        val evaluator = new ClusteringEvaluator()
//
//        val silhouette = evaluator.evaluate(distanceDF)
//        print(silhouette)

//        val chi = ChiSquareTest.test(predictions, "features", "prediction").head
//        println(s"pValues = ${chi.getAs[Vector](0)}")
//        println(s"degreesOfFreedom ${chi.getSeq[Int](1).mkString("[", ",", "]")}")
//        println(s"statistics ${chi.getAs[Vector](2)}")
//
//

//        val outliers0 = MyUtility.quantiles(spark, cluster0DF)
//        outliers0.foreach(row => println(row))
//        val outliers1 = MyUtility.quantiles(spark, cluster1DF)
//        outliers1.foreach(row => println(row))
//        val outliers2 = MyUtility.quantiles(spark, cluster2DF)
//        outliers2.foreach(row => println(row))
//        val outliers3 = MyUtility.quantiles(spark, cluster3DF)
//        outliers3.foreach(row => println(row))
//        val outliers4 = MyUtility.quantiles(spark, cluster4DF)
//        outliers4.foreach(row => println(row))

        val duration = (System.nanoTime - t1) / 1e9d
        println("Time duration: ", duration)
    }

}
