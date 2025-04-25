ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "3.7.0-RC1"

lazy val root = (project in file("."))
  .settings(
    name := "shapeful",
    resolvers ++= Resolver.sonatypeOssRepos("snapshots"),
    libraryDependencies ++= Seq(
      "dev.storch" %% "core" % "0.2.1-1.15.1",
      "dev.storch"%% "vision" %"0.2.1-1.15.1",
            "org.bytedeco" % "pytorch-platform" % "2.5.1-1.5.12-SNAPSHOT",
      "org.bytedeco" % "pytorch-platform-gpu" % "2.5.1-1.5.12-SNAPSHOT",
      "org.bytedeco" % "cuda" % "12.6-9.5-1.5.12-SNAPSHOT" classifier "linux-x86_64-redist",
      "org.bytedeco" % "openblas-platform" % "0.3.28-1.5.11",
      "org.scalameta" %% "munit" % "1.0.4" % Test
    ),
    fork := true,

  )
