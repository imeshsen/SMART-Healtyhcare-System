import axios from "axios";
import React, { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import "bootstrap/dist/css/bootstrap.min.css";
import Navbar from "./Navbar";
import image from "../assets/image4.jpg";

export default function User() {
  const [patient, setPatient] = useState({
    name: "",
    nic: "",
    mobile: "",
    hometown: "",
  });

  const { nic } = useParams();

  useEffect(() => {
    loadDetails();
  }, []);

  const loadDetails = async () => {
    const result = await axios.get(
      `http://localhost:8080/api/v1/patient/search/${nic}`
    );
    setPatient(result.data);
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
      <div className="container mt-5 pt-5">
        <div className="row">
          <div className="col-md-6 offset-md-3 border rounded pt-4 mt-5 shadow">
            <h2 className="text-center m-4  mt-5">Patient Details</h2>
            <div className="card pt-5 mt-5">
              <div className="card-header text-center">
                Hello {patient.name}!
                <ul className="list-group list-group-fluh">
                  <li className="list-group-item">
                    <b>Name: {patient.name}</b>
                  </li>
                  <li className="list-group-item">
                    <b>NIC: {patient.nic}</b>
                  </li>
                  <li className="list-group-item">
                    <b>Mobile: {patient.mobile}</b>
                  </li>
                  <li className="list-group-item">
                    <b>Hometown: {patient.hometown}</b>
                  </li>
                </ul>
              </div>
              <Link className="btn btn-primary my-2 mx-6" to={"/disease"}>
                Proceed to Disease section
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
