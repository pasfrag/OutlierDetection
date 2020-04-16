import org.apache.log4j.{Level, LogManager}
import org.apache.spark.SparkContext
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.ml.feature.{MinMaxScaler, VectorAssembler}
import org.apache.spark.ml.clustering.{BisectingKMeans, KMeans}
import org.apache.spark.ml.linalg.{Vectors, Vector}


object main {

    def main(args: Array[String]): Unit = {

        val spark: SparkSession = SparkSession.builder().config("spark.master", "local[*]").getOrCreate()
        val sc: SparkContext = spark.sparkContext
        val sqlContext: SQLContext = spark.sqlContext
        LogManager.getRootLogger.setLevel(Level.ERROR)
        import sqlContext.implicits._

        val dataDF = spark.read.option("header", "false").csv("data\\data-example.txt")

        // -------------------------------------Preprocessing-----------------------------------------------------------
        val doubleDF = dataDF
          .withColumn("x", col("_c0").cast("Double"))
          .withColumn("y", col("_c1").cast("Double"))
          .select("x", "y")

        val filteredDF = doubleDF.filter(row => (row(0) != null && row(1) != null))
//        filteredDF.show(7653)

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

        val model = new KMeans().setK(5).setSeed(1L).setMaxIter(1000).fit(output)//new BisectingKMeans().setK(5).setSeed(1L).setMaxIter(200).fit(output)
        val predictions = model.transform(output).select("x", "y", "prediction", "features")

//        predictions.show(100, truncate=false)
        predictions.select("x", "y", "prediction").write.mode("overwrite").csv("predictions.csv")
        val cluster_centers = model.clusterCenters
//        cluster_centers.foreach(println)

        // --------------------------------------------Area 51----------------------------------------------------------
        val distFromCenter = udf((features: Vector, c: Int) => Vectors.sqdist(features, model.clusterCenters(c)))

        val distanceDF = predictions.withColumn("distance", distFromCenter(predictions("features"), predictions("prediction")))
//        distanceDF.show(20, truncate = false)

        val cluster0DF = distanceDF.filter(row => row(2) == 0).orderBy($"distance".desc)
        val cluster1DF = distanceDF.filter(row => row(2) == 1).orderBy($"distance".desc)
        val cluster2DF = distanceDF.filter(row => row(2) == 2).orderBy($"distance".desc)
        val cluster3DF = distanceDF.filter(row => row(2) == 3).orderBy($"distance".desc)
        val cluster4DF = distanceDF.filter(row => row(2) == 4).orderBy($"distance".desc)

        cluster0DF.show(truncate = false)
        cluster1DF.show(truncate = false)
        cluster2DF.show(truncate = false)
        cluster3DF.show(truncate = false)
        cluster4DF.show(truncate = false)
    }

}
