import axios from "axios";
import React, { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import Navbar from "./Navbar";
import image from "../assets/image4.jpg";

export default function Selecteddisease() {
  const [disease, setDisease] = useState({
    name: "",
    type: "",
    specialist: "",
  });

  const { value } = useParams();

  useEffect(() => {
    loadDetails();
  }, []);

  const loadDetails = async () => {
    const result = await axios.get(
      `http://localhost:8080/api/v1/diseases/find/${value}`
    );
    setDisease(result.data);
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
      <div className="container mt-5">
        <div className="row mt-5">
          <div className="col-md-6 offset-md-3 border rounded p-4 mt-5 shadow">
            <h2 className="text-center m-4 mt-5">Disease Details</h2>
            <div className="card">
              <div className="card-header text-center">
                Disease: {disease.name}
                <ul className="list-group list-group-fluh">
                  <li className="list-group-item">
                    <b>Name: {disease.name}</b>
                  </li>
                  <li className="list-group-item">
                    <b>Type: {disease.type}</b>
                  </li>
                  <li className="list-group-item">
                    <b>Specialist: {disease.specialist}</b>
                  </li>
                </ul>
              </div>
              <Link className="btn btn-primary my-2 mx-6" to={"/disease"}>
                Proceed to Disease section
              </Link>

              <Link
                className="btn btn-success my-2 mx-6"
                to={`/contact/${disease.specialist}`}
              >
                Find your Doctor
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
