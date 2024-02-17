import React, { useEffect, useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import { Link } from "react-router-dom";
import axios from "axios";
import Navbar from "./Navbar";
import image from "../assets/image4.jpg";

export default function Disease() {
  const [diseases, setDiseases] = useState([]);

  const [value, setValue] = useState("");

  useEffect(() => {
    loadDiseaseNames();
  }, []);

  const loadDiseaseNames = async () => {
    const names = await axios.get(
      `http://localhost:8080/api/v1/diseases/findall`
    );
    setDiseases(names.data);
    console.log(names);
  };

  return (
    <div
      style={{
        backgroundImage: `url(${image})`,
        height: "100vh",
        fontSize: "15px",
        backgroundSize: "cover",
        backgroundRepeat: "no-repeat",
      }}
    >
      <Navbar />
      <div className="container pt-5 mt-5">
        <div className="row pt-5">
          <div className="col-md-6 offset-md-3 border rounded p-4 mt-5 pt-5 shadow text-center">
            <h2 className="text-center m-4">Choose your Disease</h2>
            <select
              className="form-control"
              value={value}
              onChange={(e) => {
                setValue(e.target.value);
              }}
            >
              <option value="">Choose Name</option>

              {diseases.map((disease) => (
                <option key={disease.name}>{disease.name}</option>
              ))}
            </select>
            <Link
              className="btn btn-primary my-2 mx-6 text-right"
              style={{ width: "100px" }}
              to={`/selected/${value}`}
            >
              Go
            </Link>
            <h3 className="text-center m-4">Or</h3>
            <Link
              type="button"
              class="btn btn-success my-2 mx-6"
              style={{ width: "400px" }}
              to={"/list"}
            >
              View full list
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
