import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions.{avg, monotonically_increasing_id, stddev_pop, udf}

object MyUtility {

    def quantiles(spark: SparkSession, df: DataFrame): DataFrame ={
        import spark.implicits._
        val quantiles = df.stat.approxQuantile("distance", Array(0.0, 0.75), 0.0)

        val Q1 = quantiles(0)
        val Q3 = quantiles(1)
        val IQR = Q3 - Q1

        val upperRange = Q3+ 1.5*IQR
        val outliers = df.filter($"distance" > upperRange)
        outliers
    }

    def neighborDensity(spark: SparkSession, df: DataFrame): DataFrame ={

        import spark.implicits._
        val size: Int = (df.count()/1000).toInt
        val std =  df.agg(stddev_pop("distance")).head().getDouble(0)/10
//        println(std)

        val distFromPoint = udf((features: Vector, point: Vector) => Vectors.sqdist(features, point))

        val clusterLimited = df.limit(size)
          .withColumnRenamed("features", "features1")
          .withColumn("id",monotonically_increasing_id())
          .select("features1", "id", "x", "y")
        val joinedDF = df.crossJoin(clusterLimited.select("features1", "id"))

        val distanceJoined = joinedDF.withColumn("distance1", distFromPoint($"features", $"features1"))
          .filter($"distance1" <= std)
        val counts = distanceJoined.groupBy($"id").count()

        val outlierPoints = clusterLimited.join(counts, clusterLimited("id") === counts("id")).filter($"count" <= 10)
          .select("x", "y")

        outlierPoints
    }

    def neighborDensityND(spark: SparkSession, df: DataFrame): DataFrame ={

        import spark.implicits._
        val mAvg = df.agg(avg("distance")).head().getDouble(0)
        val std =  df.agg(stddev_pop("distance")).head().getDouble(0) //
        val size = mAvg + 3 * std

        val distFromPoint = udf((features: Vector, point: Vector) => Vectors.sqdist(features, point))

        val clusterLimited = df.filter($"distance" > size)
          .withColumnRenamed("features", "features1")
          .withColumn("id",monotonically_increasing_id())
          .select("features1", "id", "x", "y")

        val joinedDF = df.crossJoin(clusterLimited.select("features1", "id"))

        val distanceJoined = joinedDF.withColumn("distance1", distFromPoint($"features", $"features1"))
          .filter($"distance1" <= std/10)
        val counts = distanceJoined.groupBy($"id").count()

        val outlierPoints = clusterLimited.join(counts, clusterLimited("id") === counts("id")).filter($"count" <= 20)
          .select("x", "y")

        outlierPoints
    }

}
