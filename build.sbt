import ai.kien.python.Python


ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := "3.7.0"


lazy val root = (project in file("."))
  .settings(
    name := "MapTensor",
    libraryDependencies ++= Seq(
      "dev.scalapy" %% "scalapy-core" % "0.5.3",      
      "org.scalameta" %% "munit" % "1.0.0" % Test
    ),
    fork := true
  )



