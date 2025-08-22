import ai.kien.python.Python


ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := "3.7.0"

// Add resolver for snapshot dependencies
ThisBuild / resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"

lazy val root = (project in file("."))
  .settings(
    name := "MapTensor",
    libraryDependencies ++= Seq(
      "dev.scalapy" %% "scalapy-core" % "0.5.3",
      "org.scalameta" %% "munit" % "1.0.0" % Test
    ),
    fork := true
  )

// Examples subproject
lazy val examples = (project in file("examples"))
  .dependsOn(root)
  .settings(
    name := "shapeful-examples",
    // Examples use the same Scala version and dependencies as main project
    libraryDependencies ++= Seq(
      "dev.scalapy" %% "scalapy-core" % "0.5.3",
      "ch.unibas.cs.gravis" %% "scaltair" % "0.3-SNAPSHOT"
    ),
    fork := true,
    envVars := Map("PYTHONPATH" -> (baseDirectory.value.getParentFile / "src" / "python").getAbsolutePath),
    // Don't publish examples
    publish := {},
    publishLocal := {},
    publishArtifact := false,
    // Examples source directory
    Compile / scalaSource := baseDirectory.value,
    scalafmtFailOnErrors := false
  )



