ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "3.6.4"

lazy val root = (project in file("."))
  .settings(
    name := "shapeful",
    resolvers ++= Resolver.sonatypeOssRepos("snapshots"),
    libraryDependencies ++= Seq(
      "dev.storch" %% "core" % "0.0-2dfa388-SNAPSHOT",
      "dev.storch"%% "vision" %"0.0-2dfa388-SNAPSHOT",
      "org.bytedeco" % "pytorch-platform" % "2.1.2-1.5.10",
      "org.bytedeco" % "pytorch-platform-gpu" % "2.1.2-1.5.10",
      "org.bytedeco" % "cuda-platform-redist" % "12.3-8.9-1.5.10"
    ),

  )
