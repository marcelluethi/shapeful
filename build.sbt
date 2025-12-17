import ai.kien.python.Python


ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := "3.7.0"
ThisBuild / organization := "ch.contrafactus"

// Add resolver for snapshot dependencies
ThisBuild / resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"

lazy val root = (project in file("."))
  .aggregate(core, nn, examples)
  .settings(
    name := "shapeful-root",
  )

lazy val core = (project in file("core"))
  .settings(
    name := "core",
    libraryDependencies ++= Seq(
      "dev.scalapy" %% "scalapy-core" % "0.5.3",
      "org.scalameta" %% "munit" % "1.0.0" % Test
    ),
    fork := true,
  )

lazy val nn = (project in file("nn"))
  .settings(
    name := "nn",
  )
  .dependsOn(core)

// Examples subproject
lazy val examples = (project in file("examples"))
  .dependsOn(core)
  .dependsOn(nn)
  .settings(
    name := "shapeful-examples",
    // Examples use the same Scala version and dependencies as main project
    libraryDependencies ++= Seq(
      "dev.scalapy" %% "scalapy-core" % "0.5.3",
      "io.github.quafadas" %% "scautable" % "0.0.28"
    ),
    fork := true,
    envVars := Map("PYTHONPATH" -> (baseDirectory.value.getParentFile / "src" / "python").getAbsolutePath),
    // Don't publish examples
    publish := {},
    publishLocal := {},
    publishArtifact := false,
    // Examples source directory
    Compile / scalaSource := baseDirectory.value,
    Compile / resourceDirectory := baseDirectory.value / "src" / "main" / "resources",
    scalafmtFailOnErrors := false
  )



