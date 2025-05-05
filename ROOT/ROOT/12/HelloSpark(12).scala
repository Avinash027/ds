import org.apache.spark.sql.SparkSession

object HelloSpark {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("HelloSpark")
      .master("local[*]")
      .getOrCreate()

    println("Hello, Spark World!")

    spark.stop()
  }
}
