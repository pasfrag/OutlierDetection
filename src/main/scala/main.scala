import org.apache.log4j.{Level, LogManager}
import org.apache.spark.SparkContext
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.sql.functions.col


object main {

    def main(args: Array[String]): Unit = {

        val spark: SparkSession = SparkSession.builder().config("spark.master", "local[*]").getOrCreate()
        val sc: SparkContext = spark.sparkContext
        val sqlContext: SQLContext = spark.sqlContext
        LogManager.getRootLogger.setLevel(Level.ERROR)

        val dataDF = spark.read.option("header", "false").csv("C:\\Users\\pasca\\Desktop\\Projects\\Spark_Projects\\OutlierDetection\\data\\data-example.txt")

        val doubleDF = dataDF
          .withColumn("x", col("_c0").cast("Double"))
          .withColumn("y", col("_c1").cast("Double"))
          .select("x", "y")

        val filteredDF = doubleDF.filter(row => (row(0) != null && row(1) != null))
//        filteredDF.show(7653)

    }

}
