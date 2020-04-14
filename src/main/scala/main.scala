import org.apache.log4j.{Level, LogManager}
import org.apache.spark.SparkContext
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.ml.feature.{MinMaxScaler, VectorAssembler}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.linalg.Vectors
import breeze.linalg._
import breeze.plot._
import vegas.sparkExt._
import vegas._




object main {

    def main(args: Array[String]): Unit = {

        val spark: SparkSession = SparkSession.builder().config("spark.master", "local[*]").getOrCreate()
        val sc: SparkContext = spark.sparkContext
        val sqlContext: SQLContext = spark.sqlContext
        LogManager.getRootLogger.setLevel(Level.ERROR)

        val dataDF = spark.read.option("header", "false").csv("data\\data-example.txt")

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

        val scaledDF = scale.fit(semiScaledDF).transform(semiScaledDF).select("x","y", "scaled_x", "scaled_y")
          .withColumnRenamed("scaled_x", "xv")
          .withColumnRenamed("scaled_y", "yv")


        val assembler = new VectorAssembler()
          .setInputCols(Array("xv", "yv"))
          .setOutputCol("features")

        val output = assembler.transform(scaledDF)

        val model = new KMeans().setK(5).setSeed(1L).setMaxIter(200).fit(output)
        val predictions = model.transform(output).select("x", "y", "prediction")

        predictions.show(false)
//        print(predictions)
//        predictions.write.parquet("predictions.parquet")
//        val stringCol = udf( (v:Int) => "#".concat(v.toString))
//        val predictionsNew = predictions.withColumn("color", stringCol(predictions("prediction")))
//        predictionsNew.show(false)
//        predictionsNew.write.parquet("predictions.parquet")

        import spark.implicits._
        val f = Figure()
        val p = f.subplot(0)
        val x = predictions.select("x").map(r => r.getDouble(0)).collect.toList
        val y = predictions.select("y").map(r => r.getDouble(0)).collect.toList
        val color = predictions.select("prediction").map(r => r.getInt(0)).collect.toList
        p += scatter(x, y, {_=>0.01})
        p.xlabel = "x axis"
        p.ylabel = "y axis"
        f.saveas("nooooooobs.png")

    }

}
